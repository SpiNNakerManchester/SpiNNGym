//
//  bkout.c
//  BreakOut
//
//  Created by Steve Furber on 26/08/2016.
//  Copyright © 2016 Steve Furber. All rights reserved.
//
// Standard includes
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Spin 1 API includes
#include <spin1_api.h>

// Common includes
#include <debug.h>

// Front end common includes
#include <data_specification.h>
#include <simulation.h>
#include "random.h"

#include <recording.h>

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------

// Frame delay (ms)
//#define score_delay 200 //14//20

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
typedef enum
{
  REGION_SYSTEM,
  REGION_LOGIC,
  REGION_RECORDING,
  REGION_DATA,
} region_t;

typedef enum
{
  SPECIAL_EVENT_INPUT_1,
  SPECIAL_EVENT_INPUT_2,
  SPECIAL_EVENT_INPUT_3,
  SPECIAL_EVENT_INPUT_4,
  SPECIAL_EVENT_INPUT_5,
  SPECIAL_EVENT_INPUT_6,
  SPECIAL_EVENT_INPUT_7,
  SPECIAL_EVENT_INPUT_8,
} special_event_t;

typedef enum
{
  KEY_CHOICE_0  = 0x0,
  KEY_CHOICE_1  = 0x1,
//  KEY_CHOICE_2  = 0x2,
//  KEY_CHOICE_3  = 0x3,
//  KEY_CHOICE_4  = 0x4,
//  KEY_CHOICE_5  = 0x5,
//  KEY_CHOICE_6  = 0x6,
//  KEY_CHOICE_7  = 0x7,
} arm_key_t;

//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------

static uint32_t _time;

//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

const int max_number_of_inputs = 8;

uint32_t *input_sequence;

uint32_t *truth_table;

mars_kiss64_seed_t kiss_seed;

int number_of_inputs;

int rand_seed;

int32_t current_score = 0;
int32_t stochastic = 1;
int32_t rate_on = 20;
float max_fire_prob_on;
int32_t rate_off = 5;
float max_fire_prob_off;

int32_t correct_output = -1;
int32_t output_choice[2] = {0};

uint32_t score_delay;

//! How many ticks until next frame
static uint32_t tick_in_frame = 0;

//! The upper bits of the key value that model should transmit with
static uint32_t key;

//! the number of timer ticks that this model should run for before exiting.
uint32_t simulation_ticks = 0;
uint32_t score_change_count=0;

//----------------------------------------------------------------------------
// Inline functions
//----------------------------------------------------------------------------
static inline void send_spike(int input)
{
  spin1_send_mc_packet(key | (input), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "sending spike to input %d\n", input);
//  current_score++;
}

//static inline void add_no_reward()
//{
//  spin1_send_mc_packet(key | (SPECIAL_EVENT_NO_REWARD), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "No reward\n");
////  current_score--;
//}

void resume_callback() {
    recording_reset();
}

//void add_event(int i, int j, colour_t col, bool bricked)
//{
//  const uint32_t colour_bit = (col == COLOUR_BACKGROUND) ? 0 : 1;
//  const uint32_t spike_key = key | (SPECIAL_EVENT_MAX + (i << 10) + (j << 2) + (bricked<<1) + colour_bit);
//
//  spin1_send_mc_packet(spike_key, 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "%d, %d, %u, %08x\n", i, j, col, spike_key);
//}

static bool initialize(uint32_t *timer_period)
{
    io_printf(IO_BUF, "Initialise logic: started\n");

    // Get the address this core's DTCM data starts at from SRAM
    address_t address = data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(address))
    {
      return false;
    }
    /*
    simulation_initialise(
        address_t address, uint32_t expected_app_magic_number,
        uint32_t* timer_period, uint32_t *simulation_ticks_pointer,
        uint32_t *infinite_run_pointer, int sdp_packet_callback_priority,
        int dma_transfer_done_callback_priority)
    */
    // Get the timing details and set up thse simulation interface
    if (!simulation_initialise(data_specification_get_region(REGION_SYSTEM, address),
    APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
    &infinite_run, 1, NULL))
    {
      return false;
    }
    io_printf(IO_BUF, "simulation time = %u\n", simulation_ticks);


    // Read breakout region
    address_t breakout_region = data_specification_get_region(REGION_LOGIC, address);
    key = breakout_region[0];
    io_printf(IO_BUF, "\tKey=%08x\n", key);
    io_printf(IO_BUF, "\tTimer period=%d\n", *timer_period);

    //get recording region
    address_t recording_address = data_specification_get_region(
                                       REGION_RECORDING,address);
    // Setup recording
    uint32_t recording_flags = 0;
    if (!recording_initialize(recording_address, &recording_flags))
    {
       rt_error(RTE_SWERR);
       return false;
    }

    address_t logic_region = data_specification_get_region(REGION_DATA, address);
    score_delay = logic_region[0];
    number_of_inputs = logic_region[1];
//    rand_seed = logic_region[2];
    kiss_seed[0] = logic_region[2];
    kiss_seed[1] = logic_region[3];
    kiss_seed[2] = logic_region[4];
    kiss_seed[3] = logic_region[5];
    rate_on = logic_region[6];
    rate_off = logic_region[7];
    max_fire_prob_on = (float)rate_on / 1000.f;
    max_fire_prob_off = (float)rate_off / 1000.f;
    stochastic = logic_region[8];
    input_sequence = (uint32_t *)&logic_region[9];
    truth_table = (uint32_t *)&logic_region[9 + number_of_inputs];
//    double arm_probabilities[10] = {0}
//    for (int i=1, i<number_of_inputs, i=i+1){
//        io_printf(IO_BUF, "converting arm prob %d, stage \n", temp_arm_probabilities[i] i)
//        arm_probabilities[i] = (double)temp_arm_probabilities[i] / 1000.0
//        io_printf(IO_BUF, "probs after = %d\n", arm_probabilities)
//    }
    validate_mars_kiss64_seed(kiss_seed);

    int truth_table_index = 0;
    for(int i=0; i<number_of_inputs; i=i+1){
        truth_table_index = truth_table_index + (input_sequence[i] * (1 << i));
        io_printf(IO_BUF, "%d: input %u, index %u, 2 %u\n", i, input_sequence[i], truth_table_index, 1 << i);
    }
    for(int i=0; i<(1 << number_of_inputs); i=i+1){
        io_printf(IO_BUF, "t%d: %u\n", i, truth_table[i]);
    }
    correct_output = truth_table[truth_table_index];

//    srand(rand_seed);
    //TODO check this prints right, ybug read the address
    io_printf(IO_BUF, "score delay %d\n", (uint32_t *)logic_region[0]);
    io_printf(IO_BUF, "no inputs %d\n", (uint32_t *)logic_region[1]);
    io_printf(IO_BUF, "kiss seed. %d\n", (uint32_t *)logic_region[2]);
    io_printf(IO_BUF, "seed 0x%x\n", (uint32_t *)logic_region[3]);
    io_printf(IO_BUF, "seed %u\n", logic_region[3]);
    io_printf(IO_BUF, "rate on %u\n", logic_region[6]);
    io_printf(IO_BUF, "rate off %u\n", logic_region[7]);
    io_printf(IO_BUF, "stochastic %d\n", logic_region[8]);
    io_printf(IO_BUF, "input seq %u\n", logic_region[9]);
    io_printf(IO_BUF, "tt %d\n", logic_region[9 + number_of_inputs]);
    io_printf(IO_BUF, "tt+1 %d\n", logic_region[9 + number_of_inputs + 1]);
    io_printf(IO_BUF, "tt-1 %d\n", logic_region[9 + number_of_inputs - 1]);
    io_printf(IO_BUF, "correct out %d\n", correct_output);

    io_printf(IO_BUF, "Initialise: completed successfully\n");

    return true;
}

bool was_it_correct(){
    int choice = -1;
    if (output_choice[0] > output_choice[1]){
        choice = 0;
    }
    else if (output_choice[1] > output_choice[0]){
        choice = 1;
    }
//    io_printf(IO_BUF, "c0 %u, c1 %u, c %u, score %u\n", output_choice[0], output_choice[1], choice, current_score);
    if (choice == correct_output){
        current_score = current_score + 1;
    }
//    io_printf(IO_BUF, "c0 %u, c1 %u, c %u, score %u\n", output_choice[0], output_choice[1], choice, current_score);
    output_choice[0] = 0;
    output_choice[1] = 0;
}

float rand021(){
    return (float)(mars_kiss64_seed(kiss_seed) / (float)0xffffffff);
}

void did_it_fire(uint32_t time){
//    io_printf(IO_BUF, "time = %u\n", time);
//    io_printf(IO_BUF, "time off = %u\n", time % (1000 / rate_off));
//    io_printf(IO_BUF, "time on = %u\n", time % (1000 / rate_on));
    if(stochastic){
        for(int i=0; i<number_of_inputs; i=i+1){
            if(input_sequence[i] == 0){
                if(rand021() < max_fire_prob_off){
                    send_spike(i);
                }
            }
            else{
                if(rand021() < max_fire_prob_on){
                    send_spike(i);
                }
            }
        }
    }
    else{
        for(int i=0; i<number_of_inputs; i=i+1){
            if (input_sequence[i] == 0 && time % (1000 / rate_off) == 0){
                send_spike(i);
            }
            else if(input_sequence[i] == 1 && time % (1000 / rate_on) == 0){
                send_spike(i);
            }
        }
    }
}

void mc_packet_received_callback(uint keyx, uint payload)
{
    uint32_t compare;
//    int max_number_of_bits = 8;
    compare = keyx & 0x1;
//    io_printf(IO_BUF, "compare = %x\n", compare);
//    io_printf(IO_BUF, "key = %x\n", key);
//    io_printf(IO_BUF, "payload = %x\n", payload);
    use(payload);
    if(compare == KEY_CHOICE_0){
        output_choice[0]++;
    }
    else if(compare == KEY_CHOICE_1){
        output_choice[1]++;
    }
//    else if(compare == KEY_CHOICE_2){
//        output_choice[2]++;
//    }
//    else if(compare == KEY_CHOICE_3){
//        output_choice[3]++;
//    }
//    else if(compare == KEY_CHOICE_4){
//        output_choice[4]++;
//    }
//    else if(compare == KEY_CHOICE_5){
//        output_choice[5]++;
//    }
//    else if(compare == KEY_CHOICE_6){
//        output_choice[6]++;
//    }
//    else if(compare == KEY_CHOICE_7){
//        output_choice[7]++;
//    }
    else {
        io_printf(IO_BUF, "it broke key selection %d\n", key);
    }
}

void timer_callback(uint unused, uint dummy)
{
    use(unused);
    use(dummy);

    _time++;
    score_change_count++;

    if (!infinite_run && _time >= simulation_ticks)
    {
        //spin1_pause();
        recording_finalise();
        // go into pause and resume state to avoid another tick
        simulation_handle_pause_resume(resume_callback);
        //    spin1_callback_off(MC_PACKET_RECEIVED);

        io_printf(IO_BUF, "infinite_run %d; time %d\n",infinite_run, _time);
        io_printf(IO_BUF, "simulation_ticks %d\n",simulation_ticks);
        //    io_printf(IO_BUF, "key count Left %u\n", left_key_count);
        //    io_printf(IO_BUF, "key count Right %u\n", right_key_count);

        io_printf(IO_BUF, "Exiting on timer.\n");
//        simulation_handle_pause_resume(NULL);
        simulation_ready_to_read();

        _time -= 1;
        return;
    }
    // Otherwise
    else
    {
        // Increment ticks in frame counter and if this has reached frame delay
        tick_in_frame++;
        did_it_fire(score_change_count);
        if(tick_in_frame == score_delay)
        {
            was_it_correct();
            // Reset ticks in frame and update frame
            tick_in_frame = 0;
//            update_frame();
            // Update recorded score every 1s
            if(score_change_count >= 1000){
                recording_record(0, &current_score, 4);
                score_change_count = 0;
            }
        }
    }
//    io_printf(IO_BUF, "time %u\n", ticks);
//    io_printf(IO_BUF, "time %u\n", _time);
}
//-------------------------------------------------------------------------------

INT_HANDLER sark_int_han (void);


//-------------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
void c_main(void)
{
  // Load DTCM data
  uint32_t timer_period;
  if (!initialize(&timer_period))
  {
    io_printf(IO_BUF,"Error in initialisation - exiting!\n");
    rt_error(RTE_SWERR);
    return;
  }

  tick_in_frame = 0;

  // Set timer tick (in microseconds)
  io_printf(IO_BUF, "setting timer tick callback for %d microseconds\n",
              timer_period);
  spin1_set_timer_tick(timer_period);

  io_printf(IO_BUF, "simulation_ticks %d\n",simulation_ticks);

  // Register callback
  spin1_callback_on(TIMER_TICK, timer_callback, 2);
  spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);

  _time = UINT32_MAX;

  simulation_run();




}
