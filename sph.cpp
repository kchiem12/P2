#include <cstdio>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#include "vec3.hpp"
#include "io.hpp"
#include "params.hpp"
#include "state.hpp"
#include "binhash.hpp"
#include "interact.hpp"
#include "leapfrog.hpp"

/*@q
 * ====================================================================
 */

/*@T
 * \section{Initialization}
 *
 * We've hard coded the computational domain to a unit box, but we'd prefer
 * to do something more flexible for the initial distribution of fluid.
 * In particular, we define the initial geometry of the fluid in terms of an
 * {\em indicator function} that is one for points in the domain occupied
 * by fluid and zero elsewhere.  A [[domain_fun_t]] is a pointer to an
 * indicator for a domain, which is a function that takes three floats and
 * returns 0 or 1.  Two examples of indicator functions are a little box
 * of fluid in the corner of the domain and a spherical drop.
 *@c*/
typedef int (*domain_fun_t)(float, float, float);

int box_indicator(float x, float y, float z)
{
    return (x < 0.5) && (y < 0.75) && (z < 0.5);
}

int circ_indicator(float x, float y, float z)
{
    float dx = (x-0.5);
    float dy = (y-0.5);
    float dz = (z-0.5);
    float r2 = dx*dx + dy*dy + dz*dz;
    return (r2 < 0.25*0.25*0.25);
}

/*@T
 *
 * The [[place_particles]] routine fills a region (indicated by the
 * [[indicatef]] argument) with fluid particles.  The fluid particles
 * are placed at points inside the domain that lie on a regular mesh
 * with cell sizes of $h/1.3$.  This is close enough to allow the
 * particles to overlap somewhat, but not too much.
 *@c*/
sim_state_t* place_particles(sim_param_t* param, 
                             domain_fun_t indicatef)
{
    float h  = param->h;
    float hh = h/1.3;

    // Count mesh points that fall in indicated region.
    int count = 0;
    for (float x = 0; x < 1; x += hh)
        for (float y = 0; y < 1; y += hh)
        	for (float z = 0; z < 1; z += hh)
        		count += indicatef(x,y,z);

    // Populate the particle data structure
    sim_state_t* s = alloc_state(count);
    int p = 0;
    for (float x = 0; x < 1; x += hh) {
        for (float y = 0; y < 1; y += hh) {
            for (float z = 0; z < 1; z += hh) {
                if (indicatef(x,y,z)) {
                    vec3_set(s->part[p].x, x, y, z);
                    vec3_set(s->part[p].v, 0, 0, 0);
                    ++p;
                }
            }
        }
    }
    return s;    
}

/*@T
 *
 * The [[place_particle]] routine determines the initial particle
 * placement, but not the desired mass.  We want the fluid in the
 * initial configuration to exist roughly at the reference density.
 * One way to do this is to take the volume in the indicated body of
 * fluid, multiply by the mass density, and divide by the number of
 * particles; but that requires that we be able to compute the volume
 * of the fluid region.  Alternately, we can simply compute the
 * average mass density assuming each particle has mass one, then use
 * that to compute the particle mass necessary in order to achieve the
 * desired reference density.  We do this with [[normalize_mass]].
 * 
 * @c*/
void normalize_mass(sim_state_t* s, sim_param_t* param)
{
    s->mass = 1;
    hash_particles(s, param->h);
    compute_density(s, param);
    float rho0 = param->rho0;
    float rho2s = 0;
    float rhos  = 0;
    for (int i = 0; i < s->n; ++i) {
        rho2s += (s->part[i].rho)*(s->part[i].rho);
        rhos  += s->part[i].rho;
    }
    s->mass *= ( rho0*rhos / rho2s );
}

sim_state_t* init_particles(sim_param_t* param)
{
    sim_state_t* s = place_particles(param, box_indicator);
    normalize_mass(s, param);
    return s;
}

/*@T
 * \section{The [[main]] event}
 *
 * The [[main]] routine actually runs the time step loop, writing
 * out files for visualization every few steps.  For debugging
 * convenience, we use [[check_state]] before writing out frames,
 * just so that we don't spend a lot of time on a simulation that
 * has gone berserk.
 *@c*/

void check_state(sim_state_t* s)
{
    for (int i = 0; i < s->n; ++i) {
        float xi = s->part[i].x[0];
        float yi = s->part[i].x[1];
        float zi = s->part[i].x[2];
        assert( xi >= 0 || xi <= 1 );
        assert( yi >= 0 || yi <= 1 );
        assert( zi >= 0 || zi <= 1 );
    }
}

int main(int argc, char** argv)
{   

    sim_param_t params;
    if (get_params(argc, argv, &params) != 0)
        exit(-1);
    sim_state_t* state = init_particles(&params);
    FILE* fp    = std::fopen(params.fname.c_str(), "w");
    int nframes = params.nframes;
    int npframe = params.npframe;
    float dt    = params.dt;
    int n       = state->n;

    double t_start = omp_get_wtime();
    //write_header(fp, n);
    write_header(fp, n, nframes, params.h);
    write_frame_data(fp, n, state, NULL);
    compute_accel(state, &params);
    leapfrog_start(state, dt);
    check_state(state);
    for (int frame = 1; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            compute_accel(state, &params);
            leapfrog_step(state, dt);
            check_state(state);
        }
        printf("Frame: %d of %d - %2.1f%%\n",frame, nframes, 
               100*(float)frame/nframes);
        write_frame_data(fp, n, state, NULL);
    }
    double t_end = omp_get_wtime();
    printf("Ran in %g seconds\n", t_end-t_start);

    fclose(fp);

    free_state(state);
}
