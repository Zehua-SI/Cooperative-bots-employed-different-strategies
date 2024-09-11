#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>

using namespace std;

// Define constants
#define L 100               // Size of the lattice (L x L)
#define N L*L               // Total number of players
#define RANDOMIZE 3145215   // Seed for random number generator
#define str_num 2           // Number of strategies (cooperate or defect)
#define neig_num 8          // Number of neighbors for each player
#define TOTAL_STEPS 10000   // Total simulation steps
#define COLLECT_STEPS 10000 // Number of steps to collect data at the end

// Global variables
int neighbors[N][neig_num], strategy[N]; // Store neighbors and strategies for each player
double payoff_matrix[str_num][str_num];  // Payoff matrix for the game
double sv[N];                            // Strategy values for players
int type[N];                             // Type of player (0 for ordinary, 1 for bots)

// Game parameters
double K = 0.1;    // Imitation intensity
double r = 0.1;    // Dilemma strength
double rho = 0.5;  // Proportion of bots
double theta = 0.9; // Probability of choosing cooperation for bots

// Structure to store and process category data
struct CategoryData {
    vector<int> counts; // Counts for each interval

    // Constructor: Initialize counts vector with 21 zeros (for intervals from -1 to 9, step 0.5)
    CategoryData() : counts(21, 0) {}

    // Collect data: increment count for the appropriate interval
    void collect(double payoff) {
        int index = int((payoff + 1) * 2); // Convert payoff to index, starting from -1, step 0.5
        index = max(0, min(index, 20)); // Ensure index is between 0 and 20
        counts[index]++;
    }

    // Calculate frequencies for each interval
    vector<double> get_frequencies(int total_count) {
        vector<double> freqs(21, 0.0);
        if (total_count > 0) {
            for (int i = 0; i < 21; i++) {
                freqs[i] = (double)counts[i] / total_count;
            }
        }
        return freqs;
    }
};

// Global variables for data collection
CategoryData bots, ordinary_sv_09, ordinary_sv_0;
int total_bots = 0, total_ordinary_sv_09 = 0, total_ordinary_sv_0 = 0;

// Random Number Generator module
// The following is the random number generation module.
// Use randf() to generate uniform random numbers between 0 and 1.
// Use randi(x) to generate random integers between 0 and x-1.
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

// Initialize the random number generator with an array of seeds
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

// Find neighbors for each player on the lattice
void find_neig(void)
{
    for(int i = 0 ; i < N ; i++)
    {
        neighbors[i][0] = i - L; // up 
        neighbors[i][1] = i + L; // down
        neighbors[i][2] = i - 1; // left
        neighbors[i][3] = i + 1; // right
        neighbors[i][4] = i - L - 1; // upper left 
        neighbors[i][5] = i - L + 1; // upper right
        neighbors[i][6] = i + L - 1; // down left
        neighbors[i][7] = i + L + 1; // down right

        // Handle edge cases for top and bottom rows
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
        
        // Handle edge cases for left and right columns
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

        // Handle corner cases
        if (i == 0)                  neighbors[i][4] = L*L - 1; 
        else if (i == L-1)           neighbors[i][5] = L*(L-1);  
        else if (i == L*(L-1))       neighbors[i][6] = L - 1;
        else if (i == L*L-1)         neighbors[i][7] = 0;  
    }
} 

// Initialize the game
void init_game(double r, double rho, double theta)
{
    find_neig();

    // Set up payoff matrix
    payoff_matrix[0][0] = 1;
    payoff_matrix[0][1] = (double)(-r);
    payoff_matrix[1][0] = (double)(1 + r);
    payoff_matrix[1][1] = 0;

    // Initialize the bots and ordinary players
    for (int i = 0; i < N; i++) {
        double p_i = randf();
        if (p_i < rho) type[i] = 1; // type[i]=1 bots
        else type[i] = 0;           // type[i]=0 ordinary players
    }

    // Initialize the strategy for the players and bots
    for (int i = 0; i < N; i++) {
        if (type[i] == 1) sv[i] = theta; // if player is bots, its cooperation probability (strategy value) is theta
        else sv[i] = randf();            // if player is ordinary, its cooperation probability (strategy value) is random
        
        // Compare the cooperation probability (strategy value) with random number, choose cooperation(0) or defection(1)
        if (sv[i] >= randf()) strategy[i] = 0;
        else strategy[i] = 1; 
    }
}

// Calculate payoff for a player
double cal_payoff(int x) {
    double pay = 0;
    int neig;

    // Determine the player's action based on their strategy value
    // Use 1 to denote D (defect), use 0 to denote C (cooperate)
    // This is for calculating search in the pay-off matrix
    if (sv[x] <= randf()) strategy[x] = 1;
    else strategy[x] = 0;

    // Focal player chooses its action according to the strategy value before playing with its neighbors
    for (int i = 0; i < neig_num; i++) {
        neig = neighbors[x][i]; // find neighbors
        pay += payoff_matrix[strategy[x]][strategy[neig]]; // Calculating accumulated pay-offs
    }
    return pay;
}

// Update strategy for a player based on the differences in neighbor's payoff and focal player's payoff
void learn_strategy(int x) { 
    int neig = neighbors[x][randi(neig_num)];

    double x_r = cal_payoff(x);
    double n_r = cal_payoff(neig);

    double prob = (double)1 / (1 + exp((x_r - n_r) / K));
    /*Fermi Function: K=0.1 represents the selection intensity. 
    The smaller the K, the greater the selection intensity.
    K=0.1 means that the learning probability is between 70% and 80%. 
    The reason why it's not 100% is because human society is not completely rational */
    
    // prob is the probability to imitate the neighbor's strategy, 1-prob is the probability to maintain the original strategy
    double p = randf();
    if (p < prob) sv[x] = sv[neig]; // focal player learns its neighbor's strategy value
}

// Conduct one round of the game
void round_game(void) {
    int center;
    // Asynchronous Simulation
    for (int i = 0; i < N; i++) {
        center = randi(N); // randomly select a focal player 
        if(type[center] == 0) learn_strategy(center); // learn strategy if the player is not a bot
    }
}

// Main function
int main(void) {
    sgenrand(RANDOMIZE); // Seed RNG
    init_game(r, rho, theta);
    
    for (int step = 0; step < TOTAL_STEPS; step++) {
        round_game();
        if (step >= TOTAL_STEPS - COLLECT_STEPS) {
            for (int i = 0; i < N; i++) {
                double pay = cal_payoff(i);
                if (type[i] == 1) {
                    total_bots++;
                    bots.collect(pay);
                } else {
                    if (abs(sv[i] - 0.9) < 1e-5) {
                        ordinary_sv_09.collect(pay);
                        total_ordinary_sv_09++;
                    }
                    else if (abs(sv[i] - 0.0) < 1e-2) {
                        ordinary_sv_0.collect(pay);
                        total_ordinary_sv_0++;
                    } 
                }
            }
        }
    }

    // Output results to a CSV file
    ofstream fout("Figure5_Mixed_pdf+cdf.csv");
    fout << "Category";
    for (double v = -1.0; v <= 9.0; v += 0.5) {
        fout << ", [" << v << "," << v + 0.5 << ")";
    }
    fout << "\n";

    // Lambda function to print frequencies
    auto print_freqs = [&fout](const string& name, const vector<double>& freqs) {
        fout << name;
        for (double f : freqs) {
            fout << "," << f;
        }
        fout << "\n";
    };

    // Print frequencies for each category
    print_freqs("bots", bots.get_frequencies(total_bots));
    print_freqs("ordinary sv=0.9", ordinary_sv_09.get_frequencies(total_ordinary_sv_09));
    print_freqs("ordinary sv=0.0", ordinary_sv_0.get_frequencies(total_ordinary_sv_0));

    fout.close();
    return 0;
}