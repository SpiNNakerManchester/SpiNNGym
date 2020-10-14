//
//  icub_vor_env.c
//  Environment to simulate ICub robot for particular network, see
//  https://www.overleaf.com/project/5f1ee9467b6572000190b496
//  (replace with link to document in future)
//
//  Copyright © 2020 Andrew Gait, Petrut Bogdan. All rights reserved.
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
// Enumerations
//----------------------------------------------------------------------------
// Regions where information for the environment is written
typedef enum {
  REGION_SYSTEM,
  REGION_ICUB_VOR_ENV,
  REGION_RECORDING,
  REGION_ICUB_VOR_ENV_DATA,
} region_t;

// we may need some of these at some point
//typedef enum {
//  SPECIAL_EVENT_INPUT_1,
//  SPECIAL_EVENT_INPUT_2,
//  SPECIAL_EVENT_INPUT_3,
//  SPECIAL_EVENT_INPUT_4,
//  SPECIAL_EVENT_INPUT_5,
//  SPECIAL_EVENT_INPUT_6,
//  SPECIAL_EVENT_INPUT_7,
//  SPECIAL_EVENT_INPUT_8,
//} special_event_t;

// These are the keys to be received for left/right choice
typedef enum {
  KEY_CHOICE_LEFT  = 0x0,
  KEY_CHOICE_RIGHT  = 0x1
} arm_key_t;

//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------

static uint32_t _time;

//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

//! Set global variables up here
// - head position, head velocity (from array read in to start off with?)
// - error value(s)
uint32_t error_window_size;
uint32_t number_of_inputs;
accum *head_positions;
accum *head_velocities;
accum error_value;
// I'm sure there are more to be added to this list

//! track the left and right values
uint32_t error_values[2] = {0};

//! The upper bits of the key value that model should transmit with
static uint32_t key;

//! How many ticks until next window (default size 10ms)
static uint32_t tick_in_window = 0;

//! the number of timer ticks that this model should run for before exiting.
uint32_t simulation_ticks = 0;

//----------------------------------------------------------------------------
// Inline functions
//----------------------------------------------------------------------------

// This is the function for sending a spike out from the environment, related
// to the error signal
static inline void send_spike(int input)
{
  spin1_send_mc_packet(key | (input), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "sending spike to input %d\n", input);
}

// Required if using auto-pause and resume
void resume_callback() {
    recording_reset();
}

// Initialize environment with values sent from python DSG
static bool initialize(uint32_t *timer_period)
{
    io_printf(IO_BUF, "Initialise icub_vor_env: started\n");

    // Get the address this core's DTCM data starts at from SRAM
    data_specification_metadata_t *address = data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(address)) {
      return false;
    }

    // Get the timing details and set up thse simulation interface
    if (!simulation_initialise(data_specification_get_region(REGION_SYSTEM, address),
    		APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
			&infinite_run, &_time, 1, 0)) {
      return false;
    }
    io_printf(IO_BUF, "simulation time = %u\n", simulation_ticks);


    // Read icub_vor region, which for now contains the information about keys on received mc packets
    address_t icub_vor_env_address_region = data_specification_get_region(REGION_ICUB_VOR_ENV, address);
    key = icub_vor_env_address_region[0];
    io_printf(IO_BUF, "\tKey=%08x\n", key);
    io_printf(IO_BUF, "\tTimer period=%d\n", *timer_period);

    // Get recording region
    void *recording_region = data_specification_get_region(
            REGION_RECORDING, address);

    // Setup recording
    uint32_t recording_flags = 0;
    if (!recording_initialize(&recording_region, &recording_flags)) {
       rt_error(RTE_SWERR);
       return false;
    }

    // Now get the data associated with the environment
    address_t icub_vor_env_data_region = data_specification_get_region(REGION_ICUB_VOR_ENV_DATA, address);

    // Ideally I guess this could be set up using a struct, but let's keep it simpler for now
    error_window_size = icub_vor_env_data_region[0];
    error_value = icub_vor_env_data_region[1];
    number_of_inputs = icub_vor_env_data_region[2];
    head_positions = (accum *)&icub_vor_env_data_region[3];
    head_velocities = (accum *)&icub_vor_env_data_region[3 + number_of_inputs];

    // Print values out to check
    io_printf(IO_BUF, "error_window_size %u, error_value %u\n");
    for (uint32_t i=0; i<number_of_inputs; i++) {
        io_printf(IO_BUF, "%d: position %k, velocity %k, error %k\n",
        		i, head_positions[i], head_velocities[i], error_value);
    }

    io_printf(IO_BUF, "Initialise: completed successfully\n");

    return true;
}

void update_error(uint32_t time, uint32_t index) {
    // Update the error values in here when a mc packet is received
    io_printf(IO_BUF, "At time %u, update index %u", time, index);
    error_values[index] += 1;
}

void test_the_head() {
//    io_printf(IO_BUF, "time off = %u\n", time % (1000 / rate_off));
//    io_printf(IO_BUF, "time on = %u\n", time % (1000 / rate_on));
    // This is a simplified version of what actually needs to happen
    for (uint32_t i=0; i < number_of_inputs; i++) {
        // It's here where we need to count which is bigger?
        if (error_values[0] > error_values[1]) {
            io_printf(IO_BUF, "error_values[0] %u > error_values[1] %u",
                    error_values[0], error_values[1]);
            send_spike(0);
        }
        else {
            send_spike(1);
        }
    }
}

// when a packet is received, update the error
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
        if (compare == KEY_CHOICE_LEFT) {
            // I think these are counts?
            update_error(_time, 0);
        }
        else if (compare == KEY_CHOICE_RIGHT) {
            update_error(_time, 1);
        }
        else {
            io_printf(IO_BUF, "Unexpected key value %d\n", key);
        }
    }
}

void timer_callback(uint unused, uint dummy)
{
    use(unused);
    use(dummy);

    _time++;

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
        tick_in_window++;

        // every 10ms (timesteps?) aggregate the values
        if (tick_in_window == error_window_size) {
            // Reset ticks in frame and update frame
            tick_in_window = 0;
            // Work out the error values
            test_the_head();
        }

        // There is probably a point where the error values should be reset too?

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

  tick_in_window = 0;

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
