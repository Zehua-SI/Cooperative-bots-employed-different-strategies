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
int action_list[N];          // Array to store current actions (0 for cooperate, 1 for defect) of each player
int strategy[N];             // Array to store strategies (not used in this implementation)
int type[N];                 // Type of player (0 for ordinary player, 1 for bot)

double stra_prob[N];         // Array to store strategy probabilities (cooperation tendencies) for each player
double payoff_matrix[str_num][str_num];  // Payoff matrix for the game
double rho;                  // Proportion of bots in the population
double theta = 0.9;          // Strategy value of bots (probability of choosing cooperation for bots)
double m = 1;                // Exponent for Kappa calculation
double r;                    // Dilemma strength (controls the temptation to defect)
double Kappa;                // Imitation strength (controls how likely players are to imitate others)

int num_cooperator = 0;      // Number of cooperators in the population

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
    
    num_cooperator = 0;  // Initialize cooperator count
    
    for (int i=0; i<N; i++){
        if(randf() < rho ){  // Determine if player is a bot
            type[i] = 1;  // Bot
            stra_prob[i] = theta;  // Bot's cooperation probability is theta
        }
        else{
            type[i] = 0;  // Ordinary player
            stra_prob[i] = randf();  // Random initial cooperation probability for ordinary players
        }
        
        // Determine initial action based on strategy probability
        action_list[i] = (randf() < stra_prob[i])? 0:1;  // 0 for cooperate, 1 for defect
        if(action_list[i] == 0) num_cooperator++;  // Count cooperators
    }
}

// Calculate payoff for a player
double cal_payoff(int x)
{
    int nn = N-1;  // Number of other players
    int current_c = num_cooperator;  // Current number of cooperators

    int x_act  = action_list[x];  // Action of the focal player
    if( x_act == 0 ){
        current_c--;  // Adjust cooperator count if focal player is a cooperator
    }
    
    // Calculate payoff based on the proportion of cooperators and defectors in the population
    double pay =  ( double ) current_c/nn * payoff_matrix[x_act][0] +
                  ( nn - current_c ) / nn * payoff_matrix[x_act][1];
    
    return pay;
}

// Learning process for a player
void learn_strategy(int center)
{
    int neig = randi(N);
    while (neig == center) neig=randi(N);  // Select a random different player
    
    if(type[center] == 0){  // Only ordinary players update their strategies
        double center_pay = cal_payoff(center);  // Payoff of the focal player
        double neig_pay = cal_payoff(neig);  // Payoff of the randomly selected player
        
        // Calculate probability of imitating the other player's strategy using Fermi function
        double prob = (double) 1 / ( 1 + exp( (center_pay - neig_pay)*Kappa ) );
        
        if( randf() < prob ) stra_prob[center] = stra_prob[neig];  // Imitate strategy with calculated probability
    } 
    
    // Update action based on new strategy probability
    if( action_list[center] == 0) num_cooperator--;  // Decrease cooperator count if player was cooperating
    action_list[center] = (randf() < stra_prob[center])? 0:1;  // Determine new action
    if( action_list[center] == 0) num_cooperator++;  // Increase cooperator count if player is now cooperating
}

// Simulate one round of the game
void main_process(void)
{
    int center;
    for(int i=0;i<N;i++){
        center = randi(N);  // Randomly select a focal player
        learn_strategy(center);  // Update strategy and action
    }
}

double data_out[10];  // Array to store output data

// Calculate average cooperation probability
void cal_data(){
    int nn=0; 
    double count_prob = 0.0;
    for(int i=0; i<N; i++){
        if(type[i] == 0){  // Only consider ordinary players
            nn++;
            count_prob += stra_prob[i];  // Sum cooperation probabilities
        }
    }
    
    if(nn!=0) data_out[0] = (double) count_prob/nn;  // Calculate average cooperation probability
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

    FILE *FC = fopen("Figure1C_Mixed.csv", "w");  // Open file for writing results
    fprintf(FC, "r,rho,Kappa,theta,avg_c,std_value\n");  // Header for the CSV file
    printf("r,rho,Kappa,theta,loop,c_fraction\n");  // Header for console output
    
    // Main simulation loop
    for(r = 0.00; r <= 0.21 ; r += 0.01){  // Loop over dilemma strength
        for(rho = 0.0; rho <= 0.5; rho += 0.01) {  // Loop over bot proportion
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