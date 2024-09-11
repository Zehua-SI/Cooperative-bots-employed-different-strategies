#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>

using namespace std;

// Constants for the simulation
#define L 100          // Size of one side of the square lattice
#define N L*L          // Total number of players in the lattice
#define RANDOMIZE 3145215
#define str_num 2      // Number of strategies (cooperate or defect)
#define neig_num 8     // Number of neighbors for each player

// Global variables
int neighbors[N][neig_num], strategy[N];
double payoff_matrix[str_num][str_num];
double sv[N];  // Strategy values for players
int type[N];   // Type of player (0 for ordinary player, 1 for bot)

// Simulation parameters
double Kappa;    // Imitation strength
double m;        // power of exponent
double r = 0.1;              // cdilemma strength
double rho;      // proportion of bots in a hybrid population
double theta = 0.9;    // strategy value of bots/ probability of choosing cooperation for bots

//The following is the random number generation module, use randf() to directly generate 0-1 random numbers that satisfy the uniform distribution, randi(x), generate 0---x-1 random integers
/*************************** RNG procedures ****************************************/
#define NN 624
#define MM 397
#define MATRIX_A 0x9908b0df   /* constant vector a */
#define UPPER_MASK 0x80000000 /* most significant w-r bits */
#define LOWER_MASK 0x7fffffff /* least significant r bits */
#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y)  (y >> 11)
#define TEMPERING_SHIFT_S(y)  (y << 7)
#define TEMPERING_SHIFT_T(y)  (y << 15)
#define TEMPERING_SHIFT_L(y)  (y >> 18)

static unsigned long mt[NN]; /* the array for the state vector  */
static int mti = NN + 1; /* mti==NN+1 means mt[NN] is not initialized */

// Initialize the random number generator with a seed
void sgenrand(unsigned long seed) {
    int i;
    for (i = 0; i < NN; i++) {
        mt[i] = seed & 0xffff0000;
        seed = 69069 * seed + 1;
        mt[i] |= (seed & 0xffff0000) >> 16;
        seed = 69069 * seed + 1;
    }
    mti = NN;
}

// Initialize the random number generator with an array
void lsgenrand(unsigned long seed_array[]) {
    int i;
    for (i = 0; i < NN; i++) mt[i] = seed_array[i];
    mti = NN;
}

// Generate a random number
double genrand() {
    unsigned long y;
    static unsigned long mag01[2] = { 0x0, MATRIX_A };
    if (mti >= NN) {
        int kk;
        if (mti == NN + 1) sgenrand(4357);
        for (kk = 0; kk < NN - MM; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + MM] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (; kk < NN - 1; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + (MM - NN)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt[NN - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[NN - 1] = mt[MM - 1] ^ (y >> 1) ^ mag01[y & 0x1];
        mti = 0;
    }
    y = mt[mti++];
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);
    return y;
}

// Generate a random float between 0 and 1
double randf() {
    return ((double)genrand() * 2.3283064370807974e-10);
}

// Generate a random integer between 0 and LIM-1
long randi(unsigned long LIM) {
    return((unsigned long)genrand() % LIM);
}

/********************** END of RNG ************************************/

// Function to find neighbors for each player in the lattice
void find_neig(void)
{
    for(int i = 0 ; i < N ; i++)
    {
        // Assign neighbors (up, down, left, right, and diagonals)
        neighbors[i][0] = i - L; // up 
        neighbors[i][1] = i + L; // down
        neighbors[i][2] = i - 1; // left
        neighbors[i][3] = i + 1; // right
        neighbors[i][4] = i - L - 1; // upper left 
        neighbors[i][5] = i - L + 1; // upper right
        neighbors[i][6] = i + L - 1; // down left
        neighbors[i][7] = i + L + 1; // down right

        // Handle edge cases (top, bottom, left, right edges and corners)
        if (i < L)
        {
            neighbors[i][0] = i + L * (L-1);
            neighbors[i][4] = i + L * (L-1) - 1;
            neighbors[i][5] = i + L * (L-1) + 1;
        }
        if (i > L * (L - 1) - 1)
        {
            neighbors[i][1] = i - L * (L-1);
            neighbors[i][6] = i - L * (L-1) - 1;
            neighbors[i][7] = i - L * (L-1) + 1;
        }
          
        if (i % L == 0)      
        {
            neighbors[i][2] = i + L - 1;
            neighbors[i][4] = i - 1;
            neighbors[i][6] = i + 2*L - 1; 
        }     
        if (i % L == L - 1)     
        {
            neighbors[i][3] = i - L + 1; 
            neighbors[i][5] = i - 2*L + 1; 
            neighbors[i][7] = i + 1;
        }   
        if (i == 0)                  neighbors[i][4] = L*L - 1; 
        else if (i == L-1)           neighbors[i][5] = L*(L-1);  
        else if (i == L*(L-1))       neighbors[i][6] = L - 1;
        else if (i == L*L-1)         neighbors[i][7] = 0;  
    }
} 

// Initialize the game
void init_game(double r, double rho, double m)
{
    Kappa = pow(10, m);
	find_neig();

    // Set up payoff matrix for Prisoner's Dilemma
    payoff_matrix[0][0] = 1;          // Cooperate-Cooperate，reward R
    payoff_matrix[0][1] = (double)(-r); // Cooperate-Defect，sucker's payoff， S
    payoff_matrix[1][0] = (double)(1 + r); // Defect-Cooperate，temptation， T
    payoff_matrix[1][1] = 0;          // Defect-Defect, punishment，P

    // Initialize bots and ordinary players
    for (int i = 0; i < N; i++) {
        double p_i = randf();
        if (p_i < rho) type[i] = 1;  // Bot
        else type[i] = 0;            // Ordinary player
    }

    // Initialize strategies for bots and ordinary players
    for (int i = 0; i < N; i++) {
        if (type[i] == 1) // bots
        {   
            sv[i] = theta; // the proportion of cooperative bots is theta
            if (sv[i] >= randf()) strategy[i] = 0;  // Cooperative bots
            else strategy[i] = 1;  // Defective bots
        }
        else {
            strategy[i] = randi(2);  // Random strategy （Cooperation  or Defection） for ordinary players
        } 
    }
}

// Calculate payoff for a player
double cal_payoff(int x) {
    double pay = 0;
    int neig;
    for (int i = 0; i < neig_num; i++) {
        neig = neighbors[x][i];
        pay += payoff_matrix[strategy[x]][strategy[neig]];
    }
    return pay;
}

// Learning process for a player
void learn_strategy(int x) {
    int neig = neighbors[x][randi(neig_num)];

    double x_r = cal_payoff(x);
    double n_r = cal_payoff(neig);

    // Calculate probability of imitating neighbor's strategy
    double prob = (double)1 / (1 + exp((x_r - n_r) / Kappa));
    
    double p = randf();
    if (p < prob) strategy[x] = strategy[neig];
}

// Simulate one round of the game
void round_game(void) {
    int center;
    // Asynchronous update: randomly select players to update their strategies
    for (int i = 0; i < N; i++) {
        center = randi(N); // randomly select a focal player 
        if(type[center] == 0) learn_strategy(center); // only ordinary players update strategies
    }
}

double data_out[4];  // Array to store output data

// Calculate fractions of strategies
void cal_data() {
    int x = 0, y = 0;
    int nn = 0;
    
    for (int i = 0; i < N; i++) {
        if (type[i] == 0) {  // Only consider ordinary players
            nn++;
            if (strategy[i] == 0) x++;      // Cooperators
            else if (strategy[i] == 1) y++; // Defectors
        } 
    }
    data_out[0] = (double)x / nn;  // Fraction of cooperators
    data_out[1] = (double)y / nn;  // Fraction of defectors
}

#define loop 100 // 100 loops
double record_loop[loop][2];

double mean_temp_c, mean_temp_d, dev_temp_c, dev_temp_d;

// Calculate mean and standard deviation of strategy fractions
void cal_dev(void) {
    // Calculate means
    mean_temp_c = 0;
    mean_temp_d = 0;
    for(int i = 0; i < loop; i++) {
        mean_temp_c += record_loop[i][0];
        mean_temp_d += record_loop[i][1];
    }
    mean_temp_c /= (double) loop;
    mean_temp_d /= (double) loop;
    
    // Calculate standard deviations
    dev_temp_c = 0;
    dev_temp_d = 0;
    for(int i = 0; i < loop; i++) {
        double c_part1 = (double) record_loop[i][0] - mean_temp_c;
        double c_part2 = (double) pow(c_part1, 2);
        dev_temp_c += (double) 1/loop * c_part2;
        
        double d_part1 = (double) record_loop[i][1] - mean_temp_d;
        double d_part2 = (double) pow(d_part1, 2);
        dev_temp_d += (double) 1/loop * d_part2;
    }
    
    dev_temp_c = (double)pow(dev_temp_c, 0.5);
    dev_temp_d = (double)pow(dev_temp_d, 0.5);
}

int main(void) {
    int Round = 10000; // 10000 game rounds
    sgenrand(time(0));  // Set random seed based on current time

    printf("*****start*****\n");

    FILE *Fc = fopen("Figure4A_Discrete strategy.csv", "w");
    
    // Main simulation loop
    for (m = -6; m < 4; m += 1) {
    	for(rho = 0.00; rho < 0.51; rho += 0.01){
	        for(int lo = 0; lo < loop; lo++) {
	            init_game(r, rho, m);
	            double c_temp = 0, d_temp = 0;
	            
	            // Run the simulation for 'Round=100' iterations
	            for (int i = 0; i < Round; i++) {
	                round_game();
	                cal_data();
	                // Record data for the last 2000 rounds
	                if (i >= Round - 2000) {
	                    c_temp = c_temp + data_out[0];
	                    d_temp = d_temp + data_out[1];
	                }  
	            }
	            record_loop[lo][0] = (double)c_temp / 2000;
	            record_loop[lo][1] = (double)d_temp / 2000;
	            
	            printf("%f\t%f\t%f\t%f\n", Kappa, rho, (double)c_temp/2000, (double)d_temp/2000);
	        }
        
        cal_dev();
        fprintf(Fc, "%f,%f,%f,%f,%f,%f\n", Kappa, rho, mean_temp_c, mean_temp_d, dev_temp_c, dev_temp_d);
    }
    }

    fclose(Fc);
    printf("*****done*****");
    return 0;
}