#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>

using namespace std;
// Constants for the simulation
#define L 100          // Size of one side of the square lattice (not used in this well-mixed population model)
#define N 10000        // Total number of players in the population
#define RANDOMIZE 3145215  // Random seed constant
#define str_num 2      // Number of strategies (cooperate or defect)
#define neig_num 8     // Number of neighbors for each player (not used in this well-mixed population model)

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
static int mti=NN+1; /* mti==NN+1 means mt[NN] is not initialized */
void sgenrand(unsigned long seed)
{int i;
 for (i=0;i<NN;i++) {mt[i] = seed & 0xffff0000; seed = 69069 * seed + 1;
                     mt[i] |= (seed & 0xffff0000) >> 16; seed = 69069 * seed + 1;
  }
  mti = NN;
}
void lsgenrand(unsigned long seed_array[])
{ int i; for (i=0;i<NN;i++) mt[i] = seed_array[i]; mti=NN; }
double genrand() 
{
    unsigned long y;
    static unsigned long mag01[2]={0x0, MATRIX_A};
    if (mti >= NN) 
    {
        int kk;
        if (mti == NN+1) sgenrand(4357); 
        for (kk=0;kk<NN-MM;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+MM] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (;kk<NN-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(MM-NN)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt[NN-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[NN-1] = mt[MM-1] ^ (y >> 1) ^ mag01[y & 0x1];
        mti = 0;
    }  
    y = mt[mti++]; y ^= TEMPERING_SHIFT_U(y); y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C; y ^= TEMPERING_SHIFT_L(y);
    return y;  
}

double randf(){ return ( (double)genrand() * 2.3283064370807974e-10 ); }
long randi(unsigned long LIM){ return((unsigned long)genrand() % LIM); }

/********************** END of RNG ************************************/

// Global variables
int neighbors[N][neig_num];  // Array to store neighbors for each player (not used in this well-mixed population model)
int strategy_tabel[N][N];    // Strategy table (not used in this implementation)
int strategy[N];             // Array to store current strategy (not used in this continuous strategy model)
int type[N];                 // Type of player (0 for ordinary player, 1 for bot)

double stra_prob[N];         // Array to store strategy probabilities (cooperation tendencies) for each player
double payoff_matrix[str_num][str_num];  // Payoff matrix for the game
double rho;                  // Proportion of bots in the population
double theta;          // Strategy value of bots (probability of choosing cooperation for bots)
double m = 1;                // Exponent for Kappa calculation
double r = 0.2;                    // Dilemma strength (controls the temptation to defect)
double Kappa;                // Imitation strength (controls how likely players are to imitate others)

double sum_cooperat = 0;     // Sum of all cooperation probabilities in the population

// Initialize the game
void init_game(double r, double rho, double m)
{
    Kappa = pow(10,m);  // Calculate Kappa based on m
    
    // Initialize the payoff matrix
    double temp_m[2][2]={
        {1.0,  -r},    // Reward, Sucker's payoff
        {1.0+r, 0.0}   // Temptation, Punishment
    };
    memcpy(payoff_matrix, temp_m, sizeof(temp_m));
    
    sum_cooperat = 0;  // Initialize sum of cooperation probabilities
    for (int i=0; i<N; i++){
        if(randf() < rho ){  // Determine if player is a bot
            type[i] = 1;  // Bot
            stra_prob[i] = theta;  // Bot's cooperation probability is theta
        }
        else{
            type[i] = 0;  // Ordinary player
            stra_prob[i] = randf();  // Random initial cooperation probability for ordinary players
        }
        sum_cooperat += stra_prob[i];  // Add to total cooperation probability
    }
}

// Calculate payoff for a player
double cal_payoff(int x)
{
    double x_c = stra_prob[x];  // Cooperation probability of the focal player
    double mean_other_c = ( sum_cooperat - x_c ) / ( N - 1 );  // Mean cooperation probability of all other players
    
    // Calculate payoff based on the player's cooperation probability and the mean cooperation of others
    double pay = -r * x_c + (1.0 + r) * mean_other_c;
    return pay;
}

// Learning process for a player
void learn_strategy(int center)
{
    int neig = randi(N);
    while (neig == center) neig=randi(N);  // Select a random different player
    
    double center_pay =  cal_payoff(center);  // Payoff of the focal player
    double neig_pay = cal_payoff(neig);  // Payoff of the randomly selected player
    
    // Calculate probability of imitating the other player's strategy using Fermi function
    double prob = (double) 1 / ( 1 + exp( (center_pay - neig_pay)*Kappa ) );
    
    sum_cooperat -= stra_prob[center];  // Remove current cooperation probability from sum
    if( randf() < prob ) stra_prob[center] = stra_prob[neig];  // Imitate strategy with calculated probability
    sum_cooperat += stra_prob[center];  // Add new cooperation probability to sum
}

// Simulate one round of the game
void main_process(void)
{
    int center;
    for(int i=0;i<N;i++){
        center = randi(N);  // Randomly select a focal player
        if(type[center] == 0) learn_strategy(center);  // Only ordinary players update strategies
    }
}

double data_out[10];  // Array to store output data

// Calculate average cooperation probability
void cal_data(){
    int nn=0; 
    double sv_avg = 0;
    for(int i=0; i<N; i++){
        if(type[i] == 0){  // Only consider ordinary players
            nn++;
            sv_avg += stra_prob[i];  // Sum cooperation probabilities
        }
    }
    
    if(nn!=0) data_out[0] = (double) sv_avg/nn;  // Calculate average cooperation probability
    else data_out[0] = 0.0;	
}

double accu_data[7];  // Array to store accumulated data
double avg_data[7];   // Array to store average data
int loop = 100;  // Number of loops for each parameter set

int main(void) 
{
    sgenrand(time(0));  // Set random seed based on current time

    int rd = 10000;  // Number of rounds for each simulation
    int last_rd = 2000;  // Number of rounds to consider for data collection
    
    FILE *FC = fopen("Figure6_Continuous_r=0.2.csv", "w");  // Open file for writing results
    fprintf(FC, "r,rho,Kappa,theta,avg_c,std_value\n");  // Header for the CSV file
    printf("r,rho,Kappa,theta,loop,c_fraction\n");  // Header for console output
    
    // Main simulation loop
    for(rho = 0.00; rho <= 0.51 ; rho += 0.01){  // Loop over the proportion of bots
        for(theta = 0.0; theta <= 1.00; theta += 0.01) {  // Loop over the strategy value of bots
            double record_fc[loop]={0};  // Array to store results of each loop
            double sum_fc=0.0;  // Sum of cooperation fractions
            
            for(int lo=0; lo<loop; lo++){  // Repeat each parameter set 100 times
                init_game(r, rho, m);  // Initialize the game
                for(int t=0;t<1;t++) accu_data[t]=0.0;  // Reset accumulated data
                
                // Run the simulation for 'rd' rounds
                for(int i=0; i<rd; i++){
                    main_process();  // Play one round of the game
                    
                    // Collect data for the last 'last_rd' rounds
                    if(i> rd - last_rd-1){
                        cal_data();  // Calculate average cooperation probability
                        for(int t=0;t<1;t++) accu_data[t]+=(double)data_out[t];
                    }
                }
                double temp_value = (double) accu_data[0]/last_rd;  // Calculate average cooperation probability
                record_fc[lo] = temp_value;
                sum_fc += temp_value;
                
                // Print result for each loop
                printf("%f,%f,%f,%f,%d,%f\n", r, rho, Kappa, theta, lo, temp_value);
            }
            
            // Calculate average and standard deviation
            double variance = 0;
            double avg_acc = (double) sum_fc/loop;
            for(int lc=0; lc<loop; lc++) variance += (double) pow((record_fc[lc] - avg_acc) ,2) / loop ;
            double std_value = (double) pow(variance, 0.5);
            
            // Write results to file
            fprintf(FC,"%f,%f,%f,%f,%f,%f\n", r, rho, Kappa, theta, avg_acc, std_value);
        }
    }

    fclose(FC);  // Close the output file
    return 0; 
}