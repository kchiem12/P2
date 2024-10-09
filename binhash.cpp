#include <string.h>

#include "zmorton.hpp" 
#include "binhash.hpp"

/*@q
 * ====================================================================
 */

/*@T
 * \subsection{Spatial hashing implementation}
 * 
 * In the current implementation, we assume [[HASH_DIM]] is $2^b$,
 * so that computing a bitwise of an integer with [[HASH_DIM]] extracts
 * the $b$ lowest-order bits.  We could make [[HASH_DIM]] be something
 * other than a power of two, but we would then need to compute an integer
 * modulus or something of that sort.
 * 
 *@c*/

#define HASH_MASK (HASH_DIM-1)

unsigned particle_bucket(particle_t* p, float h)
{
    unsigned ix = p->x[0]/h;
    unsigned iy = p->x[1]/h;
    unsigned iz = p->x[2]/h;
    return zm_encode(ix & HASH_MASK, iy & HASH_MASK, iz & HASH_MASK);
}

unsigned particle_neighborhood(unsigned* buckets, particle_t* p, float h)
{
    /* BEGIN TASK */
    unsigned ix = p->x[0]/h;
    unsigned iy = p->x[1]/h;
    unsigned iz = p->x[2]/h;
    
    unsigned count = 0;

    for (int dx = -1; dx <= 1; ++dx)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dz = -1; dz <= 1; ++dz)
            {
                unsigned nx = (ix + dx) & HASH_MASK;
                unsigned ny = (iy + dy) & HASH_MASK;
                unsigned nz = (iz + dz) & HASH_MASK;

                buckets[count++] = zm_encode(nx, ny, nz);
            }
        }
    }

    return count;
    /* END TASK */
}

void hash_particles(sim_state_t* s, float h)
{
    /* BEGIN TASK */
    memset(s->hash, 0, HASH_SIZE * sizeof(particle_t*));

    for (int i = 0; i < s->n; ++i) {
        particle_t* p = &s->part[i];

        unsigned bin = particle_bucket(p, h);
        p->next = s->hash[bin];
        s->hash[bin] = p;
    }
    /* END TASK */
}
