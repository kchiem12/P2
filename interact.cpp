#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#include "vec3.hpp"
#include "zmorton.hpp"

#include "params.hpp"
#include "state.hpp"
#include "interact.hpp"
#include "binhash.hpp"

/* Define this to use the bucketing version of the code */
#define USE_BUCKETING

/*@T
 * \subsection{Density computations}
 * 
 * The formula for density is
 * \[
 *   \rho_i = \sum_j m_j W_{p6}(r_i-r_j,h)
 *          = \frac{315 m}{64 \pi h^9} \sum_{j \in N_i} (h^2 - r^2)^3.
 * \]
 * We search for neighbors of node $i$ by checking every particle,
 * which is not very efficient.  We do at least take advange of
 * the symmetry of the update ($i$ contributes to $j$ in the same
 * way that $j$ contributes to $i$).
 *@c*/

inline
void update_density(particle_t* pi, particle_t* pj, float h2, float C)
{
    float r2 = vec3_dist2(pi->x, pj->x);
    float z  = h2-r2;
    if (z > 0) {
        float rho_ij = C*z*z*z;
        pi->rho += rho_ij;
        pj->rho += rho_ij;
    }
}

void compute_density(sim_state_t* s, sim_param_t* params)
{
    int n = s->n;
    particle_t* p = s->part;
    particle_t** hash = s->hash;

    float h  = params->h;
    float h2 = h*h;
    float h3 = h2*h;
    float h9 = h3*h3*h3;
    float C  = ( 315.0/64.0/M_PI ) * s->mass / h9;

    // Clear densities
    #pragma omp for
    for (int i = 0; i < n; ++i)
        p[i].rho = 0;

    // Accumulate density info
#ifdef USE_BUCKETING
    /* BEGIN TASK */
    #pragma omp parallel
    {
        // Each thread uses a private copy of density buffer to avoid race conditions
        float local_rho[n] = {0};

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            particle_t* pi = s->part + i;
            local_rho[i] += (315.0 / 64.0 / M_PI) * s->mass / h3;

            unsigned buckets[MAX_NBR_BINS];
            unsigned num_bins = particle_neighborhood(buckets, pi, h);

            for (unsigned b = 0; b < num_bins; ++b) {
                unsigned bin_index = buckets[b];
                particle_t* pj = hash[bin_index];

                while (pj != NULL) {
                    if (pj != pi && pi < pj) {
                        float r2 = vec3_dist2(pi->x, pj->x);
                        float z = h2 - r2;
                        if (z > 0) {
                            float rho_ij = C * z * z * z;
                            local_rho[i] += rho_ij;
                            local_rho[pj - s->part] += rho_ij;
                        }
                    }
                    pj = pj->next;
                }
            }
        }

        // Combine results from each thread into the global state
        #pragma omp critical
        {
            for (int i = 0; i < n; ++i) {
                p[i].rho += local_rho[i];
            }
        }
    }
    /* END TASK */
#else
    for (int i = 0; i < n; ++i) {
        particle_t* pi = s->part+i;
        pi->rho += ( 315.0/64.0/M_PI ) * s->mass / h3;
        for (int j = i+1; j < n; ++j) {
            particle_t* pj = s->part+j;
            update_density(pi, pj, h2, C);
        }
    }
#endif
}


/*@T
 * \subsection{Computing forces}
 * 
 * The acceleration is computed by the rule
 * \[
 *   \bfa_i = \frac{1}{\rho_i} \sum_{j \in N_i} 
 *     \bff_{ij}^{\mathrm{interact}} + \bfg,
 * \]
 * where the pair interaction formula is as previously described.
 * Like [[compute_density]], the [[compute_accel]] routine takes
 * advantage of the symmetry of the interaction forces
 * ($\bff_{ij}^{\mathrm{interact}} = -\bff_{ji}^{\mathrm{interact}}$)
 * but it does a very expensive brute force search for neighbors.
 *@c*/

inline
void update_forces(particle_t* pi, particle_t* pj, float h2,
                   float rho0, float C0, float Cp, float Cv)
{
    float dx[3];
    vec3_diff(dx, pi->x, pj->x);
    float r2 = vec3_len2(dx);
    if (r2 < h2) {
        const float rhoi = pi->rho;
        const float rhoj = pj->rho;
        float q = sqrt(r2/h2);
        float u = 1-q;
        float w0 = C0 * u/rhoi/rhoj;
        float wp = w0 * Cp * (rhoi+rhoj-2*rho0) * u/q;
        float wv = w0 * Cv;
        float dv[3];
        vec3_diff(dv, pi->v, pj->v);

        // Equal and opposite pressure forces
        vec3_saxpy(pi->a,  wp, dx);
        vec3_saxpy(pj->a, -wp, dx);
        
        // Equal and opposite viscosity forces
        vec3_saxpy(pi->a,  wv, dv);
        vec3_saxpy(pj->a, -wv, dv);
    }
}

void compute_accel(sim_state_t* state, sim_param_t* params)
{
    // Unpack basic parameters
    const float h    = params->h;
    const float rho0 = params->rho0;
    const float k    = params->k;
    const float mu   = params->mu;
    const float g    = params->g;
    const float mass = state->mass;
    const float h2   = h*h;

    // Unpack system state
    particle_t* p = state->part;
    particle_t** hash = state->hash;
    int n = state->n;

    // Rehash the particles
    hash_particles(state, h);

    // Compute density and color
    compute_density(state, params);

    // Start with gravity and surface forces
    #pragma omp for
    for (int i = 0; i < n; ++i)
        vec3_set(p[i].a,  0, -g, 0);

    // Constants for interaction term
    float C0 = 45 * mass / M_PI / ( (h2)*(h2)*h );
    float Cp = k/2;
    float Cv = -mu;

    // Accumulate forces
#ifdef USE_BUCKETING
    /* BEGIN TASK */
 #pragma omp parallel
    {
        float (*local_acc)[3] = (float (*)[3])calloc(n, sizeof(float[3])); // Allocate and zero-initialize

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            particle_t* pi = p + i;

            unsigned buckets[MAX_NBR_BINS];
            unsigned num_bins = particle_neighborhood(buckets, pi, h);

            for (unsigned b = 0; b < num_bins; ++b) {
                unsigned bin_index = buckets[b];
                particle_t* pj = hash[bin_index];

                while (pj != NULL) {
                    if (pj != pi && pi < pj) {
                        float dx[3], dv[3];
                        vec3_diff(dx, pi->x, pj->x);
                        vec3_diff(dv, pi->v, pj->v);

                        float r2 = vec3_len2(dx);
                        if (r2 < h2) {
                            float q = sqrt(r2 / h2);
                            float u = 1 - q;
                            float w0 = C0 * u / (pi->rho * pj->rho);
                            float wp = w0 * Cp * (pi->rho + pj->rho - 2 * rho0) * u / q;
                            float wv = w0 * Cv;

                            // Update local accelerations
                            vec3_saxpy(local_acc[i], wp, dx);
                            vec3_saxpy(local_acc[i], wv, dv);

                            int j = pj - p; // Index of pj
                            vec3_saxpy(local_acc[j], -wp, dx);
                            vec3_saxpy(local_acc[j], -wv, dv);
                        }
                    }
                    pj = pj->next;
                }
            }
        }

        // Combine results into the global particle accelerations
        #pragma omp critical
        {
            for (int i = 0; i < n; ++i) {
                vec3_saxpy(p[i].a, 1.0, local_acc[i]);
            }
        }

        free(local_acc); // Free allocated memory
    }
    /* END TASK */
#else
    for (int i = 0; i < n; ++i) {
        particle_t* pi = p+i;
        for (int j = i+1; j < n; ++j) {
            particle_t* pj = p+j;
            update_forces(pi, pj, h2, rho0, C0, Cp, Cv);
        }
    }
#endif
}