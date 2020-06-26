//
//  store_recall.c
//  Store & Recall game
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
//#define time_period 200 //14//20

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
  SPECIAL_EVENT_VALUE_0,
  SPECIAL_EVENT_VALUE_1,
  SPECIAL_EVENT_STORE,
  SPECIAL_EVENT_RECALL,
  SPECIAL_EVENT_FORGET,
} special_event_t;

typedef enum
{
    STATE_IDLE,
    STATE_STORING,
    STATE_STORED,
    STATE_RECALLING,
    STATE_FORGET,
    STATE_SHIFT,
} current_state_t;

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

typedef union{
   uint32_t u;
   float f;
   accum a;
} uint_float_union;

//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------

static uint32_t _time;

//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

mars_kiss64_seed_t kiss_seed;

int rand_seed;

int current_score_0 = 0;
int current_score_1 = 0;
int number_of_trials = 0;
int32_t stochastic = 1;
int32_t reward = 1;
float prob_command = 1.f / 6.f;
float prob_in_change = 1.f / 2.f;
int32_t time_until_command = 0;
int32_t pop_size = 1;
int32_t rate_on = 50;
float max_fire_prob_on;
int32_t rate_off = 0;
float max_fire_prob_off;

uint_float_union temp_accum;

float current_accuracy = 0.f;
int32_t current_state = STATE_IDLE;
int32_t current_value = 0;
int32_t stored_value = 0;
int32_t chose_0 = 0;
int32_t chose_1 = 0;

uint32_t time_period;

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
static inline void spike_value(int value, int pop_index)
{
    spin1_send_mc_packet(key | ((value * pop_size) + pop_index), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "sending spike to value %d from %d\n", value, ((input * pop_size) + pop_index));
//  current_score++;
}

static inline void spike_recall(int pop_index){
    spin1_send_mc_packet(key | ((SPECIAL_EVENT_RECALL * pop_size) + pop_index), 0, NO_PAYLOAD);
}

static inline void spike_store(int pop_index){
    spin1_send_mc_packet(key | ((SPECIAL_EVENT_STORE * pop_size) + pop_index), 0, NO_PAYLOAD);
}

static inline void spike_forget(int pop_index){
	use(pop_index);
//    spin1_send_mc_packet(key | ((SPECIAL_EVENT_FORGET * pop_size) + pop_index), 0, NO_PAYLOAD);
}

void resume_callback() {
    recording_reset();
}

static bool initialize(uint32_t *timer_period)
{
    io_printf(IO_BUF, "Initialise logic: started\n");

    // Get the address this core's DTCM data starts at from SRAM
    data_specification_metadata_t *address = data_specification_get_data_address();

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
			&infinite_run, &_time, 1, 0))
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
    void *recording_region = data_specification_get_region(
            REGION_RECORDING, address);

    // Setup recording
    uint32_t recording_flags = 0;
    if (!recording_initialize(&recording_region, &recording_flags))
    {
       rt_error(RTE_SWERR);
       return false;
    }

    address_t logic_region = data_specification_get_region(REGION_DATA, address);
    time_period = logic_region[0];
    pop_size = logic_region[1];
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
    reward = logic_region[9];
    temp_accum.u = logic_region[10];
    prob_command = temp_accum.a;
    temp_accum.u = logic_region[11];
    prob_in_change = temp_accum.a;

    validate_mars_kiss64_seed(kiss_seed);

//    srand(rand_seed);
    //TODO check this prints right, ybug read the address
    io_printf(IO_BUF, "time period %d\n", time_period);
    io_printf(IO_BUF, "pop_size %d\n", pop_size);
    io_printf(IO_BUF, "kiss seed. %d\n", (uint32_t *)logic_region[2]);
    io_printf(IO_BUF, "seed 0x%x\n", (uint32_t *)logic_region[3]);
    io_printf(IO_BUF, "seed %u\n", logic_region[3]);
    io_printf(IO_BUF, "rate on %u\n", rate_on);
    io_printf(IO_BUF, "rate off %u\n", rate_off);
    io_printf(IO_BUF, "stochastic %d\n", stochastic);
    io_printf(IO_BUF, "reward %u\n", reward);
    io_printf(IO_BUF, "prob action %k\n", (accum)prob_command);
    io_printf(IO_BUF, "prob in change %k\n", (accum)prob_in_change);

    io_printf(IO_BUF, "Initialise: completed successfully\n");

    return true;
}

float rand021(){
    return (float)(mars_kiss64_seed(kiss_seed) / (float)0xffffffff);
}

void move_state(){
    current_state = (current_state + 1) % STATE_SHIFT;
}

void send_value(uint32_t time){
//    io_printf(IO_BUF, "time = %u\n", time);
//    io_printf(IO_BUF, "time off = %u\n", time % (1000 / rate_off));
//    io_printf(IO_BUF, "time on = %u\n", time % (1000 / rate_on));
    if(stochastic){
        for(int i=0; i<pop_size; i=i+1){
            if(current_value == 0){
                if(rand021() < max_fire_prob_on){
                    spike_value(0, i);
                }
            }
            else{
                if(rand021() < max_fire_prob_on){
                    spike_value(1, i);
                }
            }
        }
    }
    else{
        for(int i=0; i<pop_size; i=i+1){
            if (current_value == 0 && time % (1000 / rate_on) == 0){
                spike_value(0, i);
            }
            else if(current_value == 1 && time % (1000 / rate_on) == 0){
                spike_value(1, i);
            }
        }
    }
}

void send_store_recall_forget(uint32_t time){
    if(stochastic){
        for(int i=0; i<pop_size; i=i+1){
            if(current_state == STATE_STORING){
                if(rand021() < max_fire_prob_on){
                    spike_store(i);
                }
            }
            else if(current_state == STATE_FORGET){
                if(rand021() < max_fire_prob_on){
                    spike_forget(i);
                }
            }
            else{
                if(rand021() < max_fire_prob_on){
                    spike_recall(i);
                }
            }
        }
    }
    else{
        for(int i=0; i<pop_size; i=i+1){
            if (current_state == STATE_STORING && time % (1000 / rate_on) == 0){
                spike_store(i);
            }
            else if(current_state == STATE_RECALLING && time % (1000 / rate_on) == 0){
                spike_recall(i);
            }
            else if(current_state == STATE_FORGET && time % (1000 / rate_on) == 0){
                spike_forget(i);
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
        chose_0++;
    }
    else if(compare == KEY_CHOICE_1){
        chose_1++;
    }
    else {
        io_printf(IO_BUF, "it broke key selection %d\n", key);
    }
}

void did_it_store_correctly(){
    number_of_trials++;
    if (stored_value == 0){
        if (chose_0 > chose_1){
            current_score_0++;
        }
        else if (chose_0 < chose_1 && prob_command > 0.5){
            current_score_0--;
        }
    }
    else if (stored_value == 1){
        if(chose_0 < chose_1){
            current_score_1++;
        }
        else if (chose_0 > chose_1 && prob_command > 0.5){
            current_score_1--;
        }
    }
}

void update_state(){
    if (rand021() < prob_in_change){
        current_value = (current_value + 1) % 2;
    }
    if (current_state == STATE_RECALLING || current_state == STATE_STORING){
        if (current_state == STATE_RECALLING){
            did_it_store_correctly();
            time_until_command = 5;
        }
        else{
            time_until_command = 3;
        }
        move_state();
    }
    else if (time_until_command == 0){
        move_state();
    }
    else{
        time_until_command--;
    }
    if (current_state == STATE_STORING){
        stored_value = current_value;
//        io_printf(IO_BUF, "storing state:%u, stored:%u, time:%u\n", current_state, stored_value, _time);
    }
}

void send_state(int32_t time){
    if (current_state != STATE_RECALLING){
        send_value(time);
    }
    if (current_state == STATE_RECALLING || current_state == STATE_STORING || current_state == STATE_FORGET){
        send_store_recall_forget(time);
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
        send_state(_time);
        if(tick_in_frame == time_period)
        {
            update_state();
//            io_printf(IO_BUF, "state:%u, stored:%u, time:%u\n", current_state, stored_value, _time);
            // Reset ticks in frame and update frame
            tick_in_frame = 0;
//            update_frame();
            // Update recorded score every 1s
            if(score_change_count >= 1000){
                int progress[3] = {current_score_0, current_score_1, number_of_trials};
                current_accuracy = (float)((float)(current_score_0 + current_score_1) / (float)number_of_trials);
//                accum hold = (accum)((accum)current_score / (accum)number_of_trials);
                io_printf(IO_BUF, "accuracy:%k, current_score_0:%u, current_score_1:%u, number_of_trials:%u\n", (accum)current_accuracy, current_score_0, current_score_1, number_of_trials);
//                io_printf(IO_BUF, "state:%u, time:%u\n", current_state, _time);
                recording_record(0, &progress, 12);
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
