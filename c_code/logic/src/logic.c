# Copyright (c) 2019-2021 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

//
//  logic.c
//  Logic game
//
//  Copyright © 2019 Adam Perrett. All rights reserved.
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
typedef enum {
  REGION_SYSTEM,
  REGION_LOGIC,
  REGION_RECORDING,
  REGION_DATA,
} region_t;

typedef enum {
  SPECIAL_EVENT_INPUT_1,
  SPECIAL_EVENT_INPUT_2,
  SPECIAL_EVENT_INPUT_3,
  SPECIAL_EVENT_INPUT_4,
  SPECIAL_EVENT_INPUT_5,
  SPECIAL_EVENT_INPUT_6,
  SPECIAL_EVENT_INPUT_7,
  SPECIAL_EVENT_INPUT_8,
} special_event_t;

typedef enum {
  KEY_CHOICE_0  = 0x0,
  KEY_CHOICE_1  = 0x1
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
uint32_t score_change_count = 0;

//----------------------------------------------------------------------------
// Inline functions
//----------------------------------------------------------------------------
static inline void send_spike(int input)
{
  spin1_send_mc_packet(key | (input), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "sending spike to key, input %d %d %d\n", key, input, key | (input));
//  current_score++;
}

void resume_callback(void)
{
    recording_reset();
}

static bool initialize(uint32_t *timer_period)
{
    io_printf(IO_BUF, "Initialise logic: started\n");

    // Get the address this core's DTCM data starts at from SRAM
    data_specification_metadata_t *address = data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(address)) {
      return false;
    }

    // Get the timing details and set up the simulation interface
    if (!simulation_initialise(data_specification_get_region(REGION_SYSTEM, address),
    		APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
			&infinite_run, &_time, 1, 0)) {
      return false;
    }
    io_printf(IO_BUF, "simulation time = %u\n", simulation_ticks);


    // Read breakout region
    address_t logic_address_region = data_specification_get_region(REGION_LOGIC, address);
    key = logic_address_region[0];
    io_printf(IO_BUF, "\tKey=%08x\n", key);
    io_printf(IO_BUF, "\tTimer period=%d\n", *timer_period);

    //get recording region
    void *recording_region = data_specification_get_region(
            REGION_RECORDING, address);

    // Setup recording
    uint32_t recording_flags = 0;
    if (!recording_initialize(&recording_region, &recording_flags)) {
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
    for (int i=0; i<number_of_inputs; i++) {
        truth_table_index = truth_table_index + (input_sequence[i] * (1 << i));
        io_printf(IO_BUF, "%d: input %u, index %u, 2 %u\n",
        		i, input_sequence[i], truth_table_index, 1 << i);
    }
    for (int i=0; i<(1 << number_of_inputs); i++){
        io_printf(IO_BUF, "t%d: %u\n", i, truth_table[i]);
    }
    correct_output = truth_table[truth_table_index];

//    srand(rand_seed);
    // TODO check this prints right, ybug read the address
    io_printf(IO_BUF, "score delay %d\n", score_delay);
    io_printf(IO_BUF, "no inputs %d\n", number_of_inputs);
    io_printf(IO_BUF, "kiss seed 0 %d\n", kiss_seed[0]);
    io_printf(IO_BUF, "seed 1 %u\n", kiss_seed[1]);
    io_printf(IO_BUF, "seed 2 %u\n", kiss_seed[2]);
    io_printf(IO_BUF, "rate on %u\n", rate_on);
    io_printf(IO_BUF, "rate off %u\n", rate_off);
    io_printf(IO_BUF, "stochastic %d\n", stochastic);
    io_printf(IO_BUF, "input seq 0 %u\n", input_sequence[0]);
    io_printf(IO_BUF, "input seq 1 %u\n", input_sequence[1]);
    io_printf(IO_BUF, "tt 0 %d\n", truth_table[0]);
    io_printf(IO_BUF, "tt 1 %d\n", truth_table[1]);
    io_printf(IO_BUF, "tt 2 %d\n", truth_table[2]);
    io_printf(IO_BUF, "correct out %d\n", correct_output);

    io_printf(IO_BUF, "Initialise: completed successfully\n");

    return true;
}

void was_it_correct(void ) // TODO: probably rename this function?
{
    int choice = -1;
    if (output_choice[0] > output_choice[1]) {
        choice = 0;
    }
    else if (output_choice[1] > output_choice[0]) {
        choice = 1;
    }
//    io_printf(IO_BUF, "c0 %u, c1 %u, c %u, score %u\n",
//    		output_choice[0], output_choice[1], choice, current_score);
    if (choice == correct_output){
        current_score = current_score + 1;
    }
//    io_printf(IO_BUF, "c0 %u, c1 %u, c %u, score %u\n",
//    		output_choice[0], output_choice[1], choice, current_score);
    output_choice[0] = 0;
    output_choice[1] = 0;

}

float rand021(void)
{
    return (float)(mars_kiss64_seed(kiss_seed) / (float)0xffffffff);
}

void did_it_fire(uint32_t time)
{
//    io_printf(IO_BUF, "time off = %u\n", time % (1000 / rate_off));
//    io_printf(IO_BUF, "time on = %u\n", time % (1000 / rate_on));
    if (stochastic) {
        for (int i=0; i<number_of_inputs; i++) {
//            io_printf(IO_BUF, "stochastic, input_sequence[%u] = %u\n", i, input_sequence[i]);
            if (input_sequence[i] == 0) {
                if (rand021() < max_fire_prob_off) {
                    send_spike(i);
                }
            }
            else {
                if (rand021() < max_fire_prob_on) {
                    send_spike(i);
                }
            }
        }
    }
    else {
        for (int i=0; i<number_of_inputs; i++) {
//            io_printf(IO_BUF, " input_sequence[%u] = %u\n", i, input_sequence[i]);
            if (input_sequence[i] == 0 && time % (1000 / rate_off) == 0) {
                send_spike(i);
            }
            else if (input_sequence[i] == 1 && time % (1000 / rate_on) == 0) {
                send_spike(i);
            }
        }
    }
}

void mc_packet_received_callback(uint keyx, uint payload)
{
//    io_printf(IO_BUF, "mc_packet_received_callback");
//    io_printf(IO_BUF, "key = %x\n", keyx);
//    io_printf(IO_BUF, "payload = %x\n", payload);
    uint32_t compare;
    compare = keyx & 0x1;
//    io_printf(IO_BUF, "compare = %x\n", compare);
    // If no payload has been set, make sure the loop will run
    if (payload == 0) { payload = 1; }

    for (uint count = payload; count > 0; count--) {
        if (compare == KEY_CHOICE_0) {
            output_choice[0]++;
        }
        else if (compare == KEY_CHOICE_1) {
            output_choice[1]++;
        }
        else {
            io_printf(IO_BUF, "it broke key selection %d\n", key);
        }
    }
}

void timer_callback(uint unused, uint dummy)
{
    use(unused);
    use(dummy);

    _time++;
    score_change_count++;

    if (!infinite_run && _time >= simulation_ticks) {
        //spin1_pause();
        recording_finalise();
        // go into pause and resume state to avoid another tick
        simulation_handle_pause_resume(resume_callback);
        //    spin1_callback_off(MC_PACKET_RECEIVED);

        io_printf(IO_BUF, "infinite_run %d; time %d\n", infinite_run, _time);
        io_printf(IO_BUF, "simulation_ticks %d\n", simulation_ticks);
        //    io_printf(IO_BUF, "key count Left %u\n", left_key_count);
        //    io_printf(IO_BUF, "key count Right %u\n", right_key_count);

        io_printf(IO_BUF, "Exiting on timer.\n");
//        simulation_handle_pause_resume(NULL);
        simulation_ready_to_read();

        _time -= 1;
        return;
    }
    // Otherwise
    else {
        // Increment ticks in frame counter and if this has reached frame delay
        tick_in_frame++;
//        io_printf(IO_BUF, "tick_in_frame %d score_delay %d\n", tick_in_frame, score_delay);
        did_it_fire(score_change_count);
        if (tick_in_frame == score_delay) {
            was_it_correct();
            // Reset ticks in frame and update frame
            tick_in_frame = 0;
//            update_frame();
            // Update recorded score every 1s
            if (score_change_count >= 1000) {
                recording_record(0, &current_score, 4);
                score_change_count = 0;
            }
        }
    }
//    io_printf(IO_BUF, "time %u\n", ticks);
//    io_printf(IO_BUF, "time %u\n", _time);
}

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
void c_main(void)
{
  // Load DTCM data
  uint32_t timer_period;
  if (!initialize(&timer_period)) {
    io_printf(IO_BUF, "Error in initialisation - exiting!\n");
    rt_error(RTE_SWERR);
    return;
  }

  tick_in_frame = 0;

  // Set timer tick (in microseconds)
  io_printf(IO_BUF, "setting timer tick callback for %d microseconds\n",
              timer_period);
  spin1_set_timer_tick(timer_period);

  io_printf(IO_BUF, "simulation_ticks %d\n", simulation_ticks);

  // Register callback
  spin1_callback_on(TIMER_TICK, timer_callback, 2);
  spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);
  spin1_callback_on(MCPL_PACKET_RECEIVED, mc_packet_received_callback, -1);

  _time = UINT32_MAX;

  simulation_run();

}
