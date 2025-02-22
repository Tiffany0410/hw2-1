#include "common.h"
#include <cmath>
#include <cstring>

// Global Variables
int bins_per_side;          // number of bins per side
double bin_size;            // size of bin
int num_bins;               // size of bin
int * bin_num_particles;    // array for number of particles in each bin
double bin_size_reciprocal;
bin_t *bins;

/*
Sets up a grid of bins for particle simulation, 
assigning neighbors to each bin while excluding the bin itself. 
Also initialize particle counts within each bin, preparing for simulation calculations.
*/
void init_bins(bin_t *bins) {
    for (int i = 0; i < bins_per_side; ++i) {
        for (int j = 0; j < bins_per_side; ++j) {
            int bin_index = i * bins_per_side + j;
            bin_t &bin = bins[bin_index];
            int num_neighbors = 0;
            bin_num_particles[bin_index] = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int neighbor_i = i + di;
                    int neighbor_j = j + dj;
                    if (neighbor_i >= 0 && neighbor_i < bins_per_side &&
                        neighbor_j >= 0 && neighbor_j < bins_per_side &&
                        (di != 0 || dj != 0)) 
                    {
                        int neighbor_index = neighbor_i * bins_per_side + neighbor_j;
                        bin.neighbor_id[num_neighbors] = neighbor_index;
                        num_neighbors++;
                    }
                }
            }
            bin.num_neighbors = num_neighbors;
        }
    }
}

/*
Assign particles to bins based on their positions, 
using a reciprocal multiplication for efficiency.
Update the count of particles in each bin accordingly.
*/
void assign_to_bins(bin_t *bins, particle_t *particles, int n) {
    // use multiplication instead of division
    double bin_size_reciprocal = 1.0 / bin_size;
    // assign each particle to the appropriate bin
    for(int i = 0; i < n; i++) {
        int x = int(particles[i].x * bin_size_reciprocal);
        int y = int(particles[i].y * bin_size_reciprocal);
        int bin_index = y * bins_per_side + x;
        bins[bin_index].particle_id[bin_num_particles[bin_index]++] = i;
    }
}

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    // Initialize bins
    bins_per_side = ceil(size / (cutoff * 1.8));
    bin_size = size / bins_per_side;
    num_bins = bins_per_side * bins_per_side;
    bin_size_reciprocal = 1.0 / bin_size;
    
    bins = (bin_t*) malloc(num_bins * sizeof(bin_t));
    bin_num_particles = new int[num_bins];  // Using new for dynamic array

    init_bins(bins);

    // Assign particles to bins
    assign_to_bins(bins, parts, num_parts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Compute forces
    for (int i = 0; i < num_bins; ++i) {
        bin_t &bin = bins[i];
        for (int j = 0; j < bin_num_particles[i]; ++j) {
            particle_t &p = parts[bin.particle_id[j]];
            p.ax = p.ay = 0;
            for (int q = 0; q < bin_num_particles[i]; ++q) {
                if (q != j) {
                    apply_force(p, parts[bin.particle_id[q]]);     
                }
            }
            for (int k = 0; k < bin.num_neighbors; ++k) {
                bin_t &neighbor_bin = bins[bin.neighbor_id[k]];
                for (int m = 0; m < bin_num_particles[bin.neighbor_id[k]]; ++m) {
                    apply_force(p, parts[neighbor_bin.particle_id[m]]);     
                }
            }
        }
    }

    // Reset bin counts
    memset(bin_num_particles, 0, num_bins * sizeof(int));

    // Move particles
    for (int i = 0; i < num_parts; i++) {
        move(parts[i], size);
        int x = int(parts[i].x * bin_size_reciprocal);
        int y = int(parts[i].y * bin_size_reciprocal);
        int bin_index = y * bins_per_side + x;
        bins[bin_index].particle_id[bin_num_particles[bin_index]++] = i;
    }
}
