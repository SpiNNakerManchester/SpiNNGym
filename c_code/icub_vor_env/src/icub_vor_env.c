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

#include <stdfix-full-iso.h>

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

// we may need some of these at some point, so keeping them for now
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
} lr_key_t;

//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------

static uint32_t _time;

//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

//! Parameters set from Python code go here
// - error_window_size
// - number_of_inputs
// - head position, head velocity arrays
// - perfect eye position, perfect eye velocity arrays (unused at present)
uint32_t error_window_size;
uint32_t output_size;
uint32_t number_of_inputs;
accum *head_positions;
accum *head_velocities;
accum *perfect_eye_pos;
accum *perfect_eye_vel;

//accum *eye_pos;
//accum *eye_vel;
//accum *encoded_error_rates;

//! Global error value
accum error_value = 0.0k;

//! count left and right spikes
uint32_t spike_counters[2] = {0};

//! The upper bits of the key value that model should transmit with
static uint32_t key;

//! How many ticks until next error window (default size 10ms)
static uint32_t tick_in_error_window = 0;

//! How many ticks until end of head loop
static uint32_t tick_in_head_loop = 0;

//! the number of timer ticks that this model should run for before exiting.
uint32_t simulation_ticks = 0;

//----------------------------------------------------------------------------
// Inline functions
//----------------------------------------------------------------------------

// This is the function for sending a spike out from the environment, currently
// related to the counts at the Left and Right atoms
static inline void send_spike(int input, accum value)
{
//    uint32_t payload = bitsk(value);
//    io_printf(IO_BUF, "payload %u value %k time %u", payload, value, _time);
    // The rate value needs to be sent as a uint32_t, so convert and send
    spin1_send_mc_packet(key | (input), bitsk(value), WITH_PAYLOAD);
//    spin1_send_mc_packet(key | (input), value, WITH_PAYLOAD);
//    spin1_send_mc_packet(key | (input), 0, NO_PAYLOAD);
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


    // Read icub_vor region, which for now contains the information
    // about keys on received mc packets
    address_t icub_vor_env_address_region = data_specification_get_region(
            REGION_ICUB_VOR_ENV, address);
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
    address_t icub_vor_env_data_region = data_specification_get_region(
            REGION_ICUB_VOR_ENV_DATA, address);

    // Ideally I guess this could be set up using a struct, but let's keep it simpler for now
    error_window_size = icub_vor_env_data_region[0];
    output_size = icub_vor_env_data_region[1];
    number_of_inputs = icub_vor_env_data_region[2];
    head_positions = (accum *)&icub_vor_env_data_region[3];
    head_velocities = (accum *)&icub_vor_env_data_region[3 + number_of_inputs];
    perfect_eye_pos = (accum *)&icub_vor_env_data_region[3 + (2 * number_of_inputs)];
    perfect_eye_vel = (accum *)&icub_vor_env_data_region[3 + (3 * number_of_inputs)];

    // Print some values out to check THIS HAS BEEN CHECKED ON 1st DECEMBER 2020
//    io_printf(IO_BUF, "error_window_size %u output_size %u\n", error_window_size, output_size);
//    io_printf(IO_BUF, "At 0: head pos %k, head vel %k, eye pos %k, eye vel %k\n",
//            head_positions[0], head_velocities[0], perfect_eye_pos[0], perfect_eye_vel[0]);
//    io_printf(IO_BUF, "At 250: head pos %k, head vel %k, eye pos %k, eye vel %k\n",
//            head_positions[250], head_velocities[250], perfect_eye_pos[250], perfect_eye_vel[250]);
//    io_printf(IO_BUF, "At 500: head pos %k, head vel %k, eye pos %k, eye vel %k\n",
//            head_positions[500], head_velocities[500], perfect_eye_pos[500], perfect_eye_vel[500]);
//    io_printf(IO_BUF, "At 750: head pos %k, head vel %k, eye pos %k, eye vel %k\n",
//            head_positions[750], head_velocities[750], perfect_eye_pos[750], perfect_eye_vel[750]);

    // End of initialise
    io_printf(IO_BUF, "Initialise: completed successfully\n");

    return true;
}

void update_count(uint32_t index) {
    // Update the count values in here when a mc packet is received
//    io_printf(IO_BUF, "At time %u, update index %u \n", _time, index);
    spike_counters[index] += 1;
}

// when a packet is received, update the error
void mc_packet_received_callback(uint keyx, uint payload)
{
//    io_printf(IO_BUF, "mc_packet_received_callback");
//    io_printf(IO_BUF, "key = %x\n", keyx);
//    io_printf(IO_BUF, "payload = %x\n", payload);
    uint32_t compare;
    compare = keyx & 0x1;  // This is an odd and even check.

    // If no payload has been set, make sure the loop will run
    if (payload == 0) { payload = 1; }

    // Update the spike counters based on the key value
    for (uint count = payload; count > 0; count--) {
        if (compare == KEY_CHOICE_LEFT) {
            update_count(0);
        }
        else if (compare == KEY_CHOICE_RIGHT) {
            update_count(1);
        }
        else {
            io_printf(IO_BUF, "Unexpected key value %d\n", key);
        }
    }
}

// Test the counters for the head after this loop
void test_the_head() {
    // Here I am testing this is working by sending a spike out to
    // wherever this vertex connects to, depending on which counter is higher.
//    if (spike_counters[0] > spike_counters[1]) {
////        io_printf(IO_BUF, "spike_counters[0] %u > spike_counters[1] %u \n",
////                spike_counters[0], spike_counters[1]);
//        send_spike(0, 0);
//        // L > R so "move eye left" (blank for now while we decide how to do this)
//
//    }
//    else {
////        io_printf(IO_BUF, "spike_counters[0] %u <= spike_counters[1] %u \n",
////                spike_counters[0], spike_counters[1]);
//        send_spike(1, 0);
//        // L <= R so "move eye right" (blank for now while we decide how to do this)
//
//    }

    // Here is where the error should be calculated: for now measure the error
    // from the default eye position (middle = 0.0) and stationary velocity (0.0)
    accum DEFAULT_EYE_POS = 0.0k;
    accum DEFAULT_EYE_VEL = 0.0k;

    // Error is relative (in both cases) as the test is done based on > or < 0.0
    accum error_pos = head_positions[tick_in_head_loop] - DEFAULT_EYE_POS;
    accum error_vel = head_velocities[tick_in_head_loop] - DEFAULT_EYE_VEL;
    error_value = (error_pos + error_vel);

    // The above could easily be replaced by a comparison to the perfect eye
    // position and velocity at the current value of tick_in_head_loop, once it has
    // been worked out how the spike counters at L and R translate to head/eye movement

    // Encode the error into a series of rates which are then sent on
    accum min_rate = 2.0k;
    accum max_rate = 20.0k;

    accum mid_neuron = (accum) (output_size) * 0.5k;
    accum low_threshold = absk(error_value) * mid_neuron;
    accum up_threshold = low_threshold - mid_neuron;

    // The first 100 values in the connecting pop are agonist, then the next 100 are antagonist
    uint32_t loop_size = output_size / 2;
    for (uint32_t n=0; n < loop_size; n++) {
        accum loop_value = (accum) n;
        // Unless otherwise specified, rate values are min_rate
        accum agonist_rate = min_rate;
        accum antagonist_rate = min_rate;
        if (loop_value < up_threshold) {
            if (error_value >= 0.0k) {
                // Antagonist is max_rate
                antagonist_rate = max_rate;
            } else {
                // Agonist is max_rate
                agonist_rate = max_rate;
            }
        } else if (loop_value < low_threshold) {
            accum loop_to_up_value = loop_value - up_threshold;
            accum low_to_up_value = low_threshold - up_threshold;
//            threshold_calc = inter_value1 / inter_value2;
            accum encoded_error_rate = max_rate - (
                    (max_rate-min_rate) * (loop_to_up_value / low_to_up_value));
            if (error_value >= 0.0k) {
                // Antagonist is encoded_error_rate
                antagonist_rate = encoded_error_rate;
            } else {
                // Agonist is encoded_error_rate
                agonist_rate = encoded_error_rate;
            }
        }

        // Now send the relevant spikes to the connected SSP population
        send_spike(n, agonist_rate);
        send_spike(n+loop_size, antagonist_rate);
    }

}

void timer_callback(uint unused, uint dummy)
{
    use(unused);
    use(dummy);

    _time++;

    // If the time has run out
    if (!infinite_run && _time >= simulation_ticks) {
        //spin1_pause();
        recording_finalise();

        // go into pause and resume state to avoid another tick
        simulation_handle_pause_resume(resume_callback);

        io_printf(IO_BUF, "infinite_run %d; time %d\n", infinite_run, _time);
        io_printf(IO_BUF, "simulation_ticks %d\n", simulation_ticks);

        io_printf(IO_BUF, "Exiting on timer.\n");
        simulation_ready_to_read();

        _time -= 1;
        return;
    }
    // Otherwise
    else {
        // Increment ticks for head loop and error window
        tick_in_head_loop++;
        tick_in_error_window++;

        // Test whether we have reached the end of the head inputs
        if (tick_in_head_loop == number_of_inputs) {
            // Reset back to zero if so
            tick_in_head_loop = 0;
        }

        // If ticks_in_error_window has reached error_window_size then compare
        // the counters and calculate error
        if (tick_in_error_window == error_window_size) {
            // Check spike counters and calculate error value
            test_the_head();

            // Do recording
            recording_record(0, &spike_counters[0], 4);
            recording_record(1, &spike_counters[1], 4);
            recording_record(2, &error_value, 4);
            recording_record(3, &head_positions[tick_in_head_loop], 4);
            recording_record(4, &head_velocities[tick_in_head_loop], 4);

            // Reset ticks in error window
            tick_in_error_window = 0;

            // Reset the spike_counters
            spike_counters[0] = 0;
            spike_counters[1] = 0;
        }

    }
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

  // Initialise (probably unnecessary...)
  tick_in_error_window = 0;
  tick_in_head_loop = 0;

  // Set timer tick (in microseconds)
  io_printf(IO_BUF, "setting timer tick callback for %d microseconds\n",
              timer_period);
  spin1_set_timer_tick(timer_period);

  io_printf(IO_BUF, "simulation_ticks %d\n", simulation_ticks);

  // Register callbacks
  spin1_callback_on(TIMER_TICK, timer_callback, 2);
  spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);
  spin1_callback_on(MCPL_PACKET_RECEIVED, mc_packet_received_callback, -1);

  _time = UINT32_MAX;

  simulation_run();

}
