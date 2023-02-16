/*
 * Copyright (c) 2013 The University of Manchester
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
// Standard includes
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Spin 1 API includes
#include <spin1_api.h>

// Common includes
#include <debug.h>
#include <sincos.h>

// Front end common includes
#include <data_specification.h>
#include <simulation.h>
#include "random.h"
#include <stdfix.h>
#include <math.h>
#include <common/maths-util.h>

#include <recording.h>

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------

// Frame delay (ms)
//#define time_increment 200 //14//20
/*
    number of bins for current angle of the pole
    number of bins for the force to be applied or number of spikes per tick equals a force
    mass of the cart
    mass of the pole
    initial starting angle
    velocity of the cart
    velocity of the pendulum
    base rate for the neurons to fire in each bin
    each spike equals a change in force to be applied (what is that amount)
    receptive field of each bin
    update model on each timer tick and on each spike received,
    or number of spikes per tick equals a force

    add option to rate (increased poisson P()) code and rank code
*/

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
typedef enum {
  REGION_SYSTEM,
  REGION_PENDULUM,
  REGION_RECORDING,
  REGION_DATA,
} region_t;

typedef enum {
  SPECIAL_EVENT_ANGLE,
  SPECIAL_EVENT_ANGLE_2,
  SPECIAL_EVENT_CART,
  SPECIAL_EVENT_ANGLE_V,
  SPECIAL_EVENT_ANGLE_2_V,
  SPECIAL_EVENT_CART_V,
} special_event_t;

typedef enum {
  // forward will be considered positive motion
  BACKWARD_MOTOR  = 0x0,
  FORWARD_MOTOR  = 0x1,
} arm_key_t;

typedef union {
   uint32_t u;
//   uint32_t* us;
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

int32_t current_score = 0;
int32_t reward_based = 1;

// experimental constraints and variables
float current_time = 0;
float max_motor_force = 10; // N
float min_motor_force = -10; // N
float motor_force = 0;
float force_increment = 100;
float track_length = 4.8; // m
float cart_position = 0; // m
float cart_velocity = 0;  // m/s
float cart_acceleration = 0;  // m/s^2
float highend_cart_v = 5; // used to calculate firing rate and bins
float max_pole_angle = (0.2f) * M_PI; // (36 / 180 = 0.2)
float min_pole_angle = -(0.2f) * M_PI;
float max_pole_angle_bin = (0.2f) * M_PI;
float min_pole_angle_bin = -(0.2f) * M_PI;

uint_float_union pole_angle_accum;
float pole_angle;
float pole_velocity = 0; // angular/s
float pole_acceleration = 0; // angular/s^2
uint_float_union half_pole_length_accum; // m
float half_pole_length; // m

uint_float_union pole2_angle_accum;
float pole2_angle;
float pole2_velocity = 0; // angular/s
float pole2_acceleration = 0; // angular/s^2
uint_float_union half_pole2_length_accum; // m
float half_pole2_length; // m

float highend_pole_v = 5; // used to calculate firing rate and bins

//#define max_bins 10
//float pole_angle_spike_time[max_bins] = {0.f};
//float pole_velocity_spike_time[max_bins] = {0.f};
//float cart_position_spike_time[max_bins] = {0.f};
//float cart_velocity_spike_time[max_bins] = {0.f};

int max_firing_rate = 20;
float max_firing_prob = 0;
int encoding_scheme = 0; // 0: rate, 1: time, 2: rank (replace with type def
int number_of_bins = 20;
float bin_width;
float bin_overlap = 2.5;
float tau_force;
uint_float_union temp_accum;

// if it's central that means perfectly central on the track and angle is the
// lowest rate, else half
int central = 1;

// experimental parameters
float max_balance_time = 0;
float current_state[3];
bool in_bounds = true;

uint32_t time_increment;

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
static inline void spike_angle(int bin)
{
    uint32_t mask;
    mask = (SPECIAL_EVENT_ANGLE * number_of_bins) + bin;
    spin1_send_mc_packet(key | (mask), 0, NO_PAYLOAD);
//    io_printf(IO_BUF, "spike_angle \t%d - \t%u\n", bin, mask);
}

static inline void spike_angle_v(int bin)
{
    uint32_t mask;
    mask = (SPECIAL_EVENT_ANGLE_V * number_of_bins) + bin;
    spin1_send_mc_packet(key | (mask), 0, NO_PAYLOAD);
//    io_printf(IO_BUF, "spike_angle_v \t%d - \t%u\n", bin, mask);
}

static inline void spike_angle_2(int bin)
{
    uint32_t mask;
    mask = (SPECIAL_EVENT_ANGLE_2 * number_of_bins) + bin;
    spin1_send_mc_packet(key | (mask), 0, NO_PAYLOAD);
//    io_printf(IO_BUF, "spike_angle \t%d - \t%u\n", bin, mask);
}

static inline void spike_angle_2_v(int bin)
{
    uint32_t mask;
    mask = (SPECIAL_EVENT_ANGLE_2_V * number_of_bins) + bin;
    spin1_send_mc_packet(key | (mask), 0, NO_PAYLOAD);
//    io_printf(IO_BUF, "spike_angle_v \t%d - \t%u\n", bin, mask);
}

static inline void spike_cart(int bin)
{
    uint32_t mask;
    mask = (SPECIAL_EVENT_CART * number_of_bins) + bin;
    spin1_send_mc_packet(key | (mask), 0, NO_PAYLOAD);
//    io_printf(IO_BUF, "spike_cart \t%d - \t%u\n", bin, mask);
}

static inline void spike_cart_v(int bin)
{
    uint32_t mask;
    mask = (SPECIAL_EVENT_CART_V * number_of_bins) + bin;
    spin1_send_mc_packet(key | (mask), 0, NO_PAYLOAD);
//    io_printf(IO_BUF, "spike_cart_v \t%d - \t%u\n", bin, mask);
}

void resume_callback(void)
{
    recording_reset();
}

//void add_event(int i, int j, colour_t col, bool bricked)
//{
//  const uint32_t colour_bit = (col == COLOUR_BACKGROUND) ? 0 : 1;
//  const uint32_t spike_key = key | (
//		  SPECIAL_EVENT_MAX + (i << 10) + (j << 2) + (bricked<<1) + colour_bit);
//
//  spin1_send_mc_packet(spike_key, 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "%d, %d, %u, %08x\n", i, j, col, spike_key);
//}

static bool initialize(uint32_t *timer_period)
{
    io_printf(IO_BUF, "Initialise double inverted pendulum: started\n");

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


    // Read pendulum region
    address_t pendulum_region = data_specification_get_region(REGION_PENDULUM, address);
    key = pendulum_region[0];
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

    cart_position = track_length * 0.5f;

    address_t pend_region = data_specification_get_region(REGION_DATA, address);
//    encoding_scheme = pend_region[0]; // 0 rate
    encoding_scheme = pend_region[0];
    time_increment = pend_region[1];
    half_pole_length_accum.u = pend_region[2];
    half_pole_length = half_pole_length_accum.a * 0.5f;
    pole_angle_accum.u = pend_region[3];
    pole_angle = pole_angle_accum.a;
    pole_angle = (pole_angle / 180.0f) * M_PI;
    half_pole2_length_accum.u = pend_region[4];
    half_pole2_length = half_pole2_length_accum.a * 0.5f;
    pole2_angle_accum.u = pend_region[5];
    pole2_angle = 0.1; // pole_angle_accum.a;  // TODO: not sure what intended here
    pole2_angle = (pole2_angle / 180.0f) * M_PI;
    reward_based = pend_region[6];
    force_increment = pend_region[7]; // (float)pend_region[5] / (float)0xffff;
    max_firing_rate = pend_region[8];
    number_of_bins = pend_region[9];
    central = pend_region[10];

    bin_width = 1.f / ((float)number_of_bins - 1.f);
    max_firing_prob = max_firing_rate * 0.001f;
//    accum
    // pass in random seeds
    kiss_seed[0] = pend_region[11];
    kiss_seed[1] = pend_region[12];
    kiss_seed[2] = pend_region[13];
    kiss_seed[3] = pend_region[14];
    validate_mars_kiss64_seed(kiss_seed);

    temp_accum.u = pend_region[15];
    bin_overlap = temp_accum.a;
    temp_accum.u = pend_region[16];
    tau_force = temp_accum.a;

    force_increment = (float)((max_motor_force - min_motor_force) / (float)force_increment);

    //TODO check this prints right, ybug read the address
//    io_printf(IO_BUF, "r1 %d\n", (uint32_t *)pend_region[0]);
//    io_printf(IO_BUF, "r2 %d\n", (uint32_t *)pend_region[1]);
//    io_printf(IO_BUF, "rand3. %d\n", (uint32_t *)pend_region[2]);
//    io_printf(IO_BUF, "rand3 0x%x\n", (uint32_t *)pend_region[3]);
//    io_printf(IO_BUF, "r4 0x%x\n", pend_region[3]);
//    io_printf(IO_BUF, "r5 0x%x\n", pend_region);
//    io_printf(IO_BUF, "encode %u\n", pend_region[0]);
//    io_printf(IO_BUF, "d %d\n", pend_region[0]);
//    io_printf(IO_BUF, "increm %u\n", pend_region[1]);
//    io_printf(IO_BUF, "d %d\n", pend_region[1]);
//    io_printf(IO_BUF, "halfp %f\n", pend_region[2]);
//    io_printf(IO_BUF, "d %f\n", pend_region[2]);
//    io_printf(IO_BUF, "half %k\n", (accum)half_pole_length);
//    io_printf(IO_BUF, "d %f\n", (float)half_pole_length);
//    io_printf(IO_BUF, "half accum %k\n", half_pole_length_accum.a);
//    io_printf(IO_BUF, "d %f\n", (float)half_pole_length_accum.a);
//    io_printf(IO_BUF, "anglep %f\n", pend_region[3]);
//    io_printf(IO_BUF, "d %f\n", pend_region[3]);
//    io_printf(IO_BUF, "angle accum %k\n", pole_angle_accum.a);
//    io_printf(IO_BUF, "f %f\n", (float)pole_angle_accum.a);
//    io_printf(IO_BUF, "angle %k\n", (accum)pole_angle);
//    io_printf(IO_BUF, "f %k\n", (float)pole_angle);
//    io_printf(IO_BUF, "reward %u\n", pend_region[4]);
//    io_printf(IO_BUF, "d %d\n", pend_region[4]);
//    io_printf(IO_BUF, "force %u\n", pend_region[5]);
//    io_printf(IO_BUF, "d %d\n", pend_region[5]);
//    io_printf(IO_BUF, "max %u\n", pend_region[6]);
//    io_printf(IO_BUF, "d %d\n", pend_region[6]);
//    io_printf(IO_BUF, "bins %u\n", pend_region[7]);
//    io_printf(IO_BUF, "d %d\n", pend_region[7]);
//    io_printf(IO_BUF, "central %u\n", pend_region[8]);
//    io_printf(IO_BUF, "d %d\n", pend_region[8]);
//    io_printf(IO_BUF, "re %d\n", reward_based);
//    io_printf(IO_BUF, "r6 0x%x\n", *pend_region);
//    io_printf(IO_BUF, "r6 0x%x\n", &pend_region);

    io_printf(IO_BUF, "starting state (d,v,a) & 2:(%k, %k, %k) & (%k, %k, %k) ",
    		"and cart (d,v,a):(%k, %k, %k)\n",
			(accum)pole_angle, (accum)pole_velocity, (accum)pole_acceleration,
			(accum)pole2_angle, (accum)pole2_velocity, (accum)pole2_acceleration,
			(accum)cart_position, (accum)cart_velocity, (accum)cart_acceleration);

    io_printf(IO_BUF, "Initialise: completed successfully\n");
    log_info("End of initialise");

//    auto start = chrono::steady_clock::now();
    return true;
}

//float firing_time(float relative_value, int bin){
//    float separation = relative_value - (bin_width * (float)bin);
//    float maximum_time_window = 1000.f / max_firing_rate;
//    float delay;
//    if (separation < 1){
//        delay = maximum_time_window * separation;
//    }
//    else{
//        if (encoding_scheme == 2){
//            delay = maximum_time_window;
//        }
//        else{
//            delay = maximum_time_window * separation;
//        }
//    }
//    return delay;
//}

// updates the current state of the pendulum
bool update_state(float time_step){
    float gravity = -9.8; // m/s^2
    float mass_cart = 1; // kg
    float mass_pole_pm = 0.1; // kg
    float friction_cart_on_track = 0.0005; // coefficient of friction
    float friction_pole_hinge = 0.000002; // coefficient of friction

    accum test_a = sink((accum)pole_angle);
    float test_f = (float)sink((accum)pole_angle);
    temp_accum.a = sink((accum)pole_angle);

    io_printf(IO_BUF, "test a %k, f %k, temp a %k, temp f %k\n",
    		test_a, (accum)test_f, (accum)temp_accum.a, (accum)temp_accum.f);

    io_printf(IO_BUF, "before state (d,v,a) & 2:(%k, %k, %k) & (%k, %k, %k) ",
    		"and cart (d,v,a):(%k, %k, %k)\n",
			(accum)pole_angle, (accum)pole_velocity, (accum)pole_acceleration,
			(accum)pole2_angle, (accum)pole2_velocity, (accum)pole2_acceleration,
			(accum)cart_position, (accum)cart_velocity, (accum)cart_acceleration);

    // equation fro pole 1
    float effective_force_pole_on_cart = 0;
    float mass_pole = mass_pole_pm * half_pole_length;
    float pole_angle_force = (mass_pole * half_pole_length * pole_velocity * pole_velocity *
    		(float)sink((accum)pole_angle));
    float angle_scalar = (0.75f * mass_pole *  (float)cosk((accum)pole_angle));
    float friction_and_gravity = (((friction_pole_hinge * pole_velocity) / (
    		mass_pole * half_pole_length)) + (gravity * (float)sink((accum)pole_angle)));
    float effective_pole_mass = mass_pole * (1.0f - ((3.0f / 4.0f) *  (float)cosk(
    		(accum)pole_angle) *  (float)cosk((accum)pole_angle)));
    effective_force_pole_on_cart = pole_angle_force + (angle_scalar * friction_and_gravity);

    // equation for pole 2
    mass_pole = mass_pole_pm * half_pole2_length;
    pole_angle_force = (mass_pole * half_pole2_length * pole2_velocity * pole2_velocity *
    		(float)sink((accum)pole2_angle));
    angle_scalar = (0.75f * mass_pole *  (float)cosk((accum)pole2_angle));
    friction_and_gravity = ((friction_pole_hinge * pole2_velocity) / (
    		mass_pole * half_pole2_length)) + (gravity * (float)sink((accum)pole2_angle));
    effective_pole_mass = effective_pole_mass + (mass_pole * (
    		1.0f - ((3.0f / 4.0f) *  (float)cosk((accum)pole2_angle) *
    				(float)cosk((accum)pole2_angle))));

    effective_force_pole_on_cart = effective_force_pole_on_cart + pole_angle_force + (
    		angle_scalar * friction_and_gravity);

    if (cart_velocity > 0){
        cart_acceleration = (motor_force - friction_cart_on_track + effective_force_pole_on_cart) /
                                (mass_cart + effective_pole_mass);
    }
    else{
        cart_acceleration = (motor_force + friction_cart_on_track + effective_force_pole_on_cart) /
                                (mass_cart + effective_pole_mass);
    }
    // time step pole 1
    mass_pole = mass_pole_pm * half_pole_length;
    float length_scalar = -3.0f / (4.0f * half_pole_length);
    float cart_acceleration_effect = cart_acceleration * (float)cosk((accum)pole_angle);
    float gravity_effect = gravity * (float)sink((accum)pole_angle);
    float friction_effect = (friction_pole_hinge * pole_velocity) / (
    		mass_pole * half_pole_length);
    pole_acceleration = length_scalar * (
    		cart_acceleration_effect + gravity_effect + friction_effect);

    io_printf(IO_BUF, "half_pole_length, length_scalar, cart_effect, gravity_effect,"
    		"friction effect, pole_acceleration %k %k %k %k %k %k\n",
    		(accum)half_pole_length, (accum)length_scalar, (accum)cart_acceleration_effect,
			(accum)gravity_effect, (accum)friction_effect, (accum)pole_acceleration);

    pole_velocity = (pole_acceleration * time_step) + pole_velocity;
    pole_angle = (pole_velocity * time_step) + pole_angle;

    // time step pole 2
    mass_pole = mass_pole_pm * half_pole2_length;
    length_scalar = -3.0f / (4.0f * half_pole2_length);
    cart_acceleration_effect = cart_acceleration * (float)cosk((accum)pole2_angle);
    gravity_effect = gravity * (float)sink((accum)pole2_angle);
    friction_effect = (friction_pole_hinge * pole2_velocity) / (mass_pole * half_pole2_length);
    pole2_acceleration = length_scalar * (
    		cart_acceleration_effect + gravity_effect + friction_effect);

    pole2_velocity = (pole2_acceleration * time_step) + pole2_velocity;
    pole2_angle = (pole2_velocity * time_step) + pole2_angle;

    cart_velocity = (cart_acceleration * time_step) + cart_velocity;
    cart_position = (cart_velocity * time_step) + cart_position;

    if (tau_force){
        motor_force = motor_force * exp(time_step / tau_force);
    }
    else{
        motor_force = 0;
    }

    io_printf(IO_BUF, "after state (d,v,a) & 2:(%k, %k, %k) & (%k, %k, %k) ",
    		"and cart (d,v,a):(%k, %k, %k)\n",
			(accum)pole_angle, (accum)pole_velocity, (accum)pole_acceleration,
			(accum)pole2_angle, (accum)pole2_velocity, (accum)pole2_acceleration,
			(accum)cart_position, (accum)cart_velocity, (accum)cart_acceleration);

    if (cart_position > track_length || cart_position < 0 ||
        pole_angle > max_pole_angle  || pole_angle < min_pole_angle ||
        pole2_angle > max_pole_angle  || pole2_angle < min_pole_angle) {
        io_printf(IO_BUF, "failed out\n");
        return false;
    }
    else{
        return true;
    }
}

void mc_packet_received_callback(uint keyx, uint payload)
{
    // make this bin related for rank encoding, relate to force increments
    uint32_t compare;
    compare = keyx & 0x1;
//    io_printf(IO_BUF, "compare = %x\n", compare);
    // If no payload has been set, make sure the loop will run
    if (payload == 0) { payload = 1; }

    for (uint count = payload; count > 0; count--) {
        if (compare == BACKWARD_MOTOR) {
            motor_force = motor_force - force_increment;
            if (motor_force < min_motor_force) {
                motor_force = min_motor_force;
            }
        }
        else if (compare == FORWARD_MOTOR) {
            motor_force = motor_force + force_increment;
            if (motor_force > max_motor_force) {
                motor_force = max_motor_force;
            }
        }
    }
}

float rand021(void)
{
    return (float)(mars_kiss64_seed(kiss_seed) / (float)0xffffffff);
}

float norm_dist(float mean, float stdev)
{
    accum norm_dist;
    norm_dist = gaussian_dist_variate(mars_kiss64_seed, NULL);
    norm_dist = (norm_dist * stdev) + mean;
    return (float)norm_dist;
}

bool firing_prob(float relative_value, int bin)
{
    float norm_value = norm_dist(0, bin_width / bin_overlap);
    float separation = relative_value - (bin_width * (float)bin);
    if (separation < 0){
        separation = -separation;
    }
//    io_printf(IO_BUF, "norm = %k, separation = %k, realtive = %k, bin = %d\n",
//    		(accum)norm_value, (accum)separation, (accum)relative_value, bin);
    if (norm_value < 0){
        norm_value = -norm_value;
    }
    if (norm_value > separation){
        if (rand021() < max_firing_prob){
            return true;
        }
        else{
            return false;
        }
    }
    else{
        return false;
    }
}

void send_status(void)
{
    float relative_cart;
    float relative_cart_velocity;
    float relative_angle;
    float relative_angular_velocity;
    float relative_angle_2;
    float relative_angular_velocity_2;
    relative_cart = cart_position / track_length;
//    io_printf(IO_BUF, "rela = %k, cart = %k, max = %k\n",
//    		(accum)relative_cart, (accum)cart_position, (accum)track_length);
    relative_cart_velocity = (cart_velocity + highend_cart_v) / (2.f * highend_cart_v);
//    io_printf(IO_BUF, "rela = %k, cartv = %k, maxv = %k\n",
//    		(accum)relative_cart_velocity, (accum)cart_velocity, (accum)highend_cart_v);
    relative_angle = (pole_angle + max_pole_angle_bin) / (2.f * max_pole_angle_bin);
//    io_printf(IO_BUF, "rela = %k, ang = %k, max = %k\n",
//    		(accum)relative_angle, (accum)pole_angle, (accum)max_pole_angle_bin);
    relative_angular_velocity = (pole_velocity + highend_pole_v) / (2.f * highend_pole_v);
//    io_printf(IO_BUF, "rela = %k, angv = %k, maxv = %k\n",
//    		(accum)relative_angular_velocity, (accum)pole_velocity, (accum)highend_pole_v);
    relative_angle_2 = (pole2_angle + max_pole_angle_bin) / (2.f * max_pole_angle_bin);
//    io_printf(IO_BUF, "rela = %k, ang = %k, max = %k\n",
//    		(accum)relative_angle, (accum)pole_angle, (accum)max_pole_angle_bin);
    relative_angular_velocity_2 = (pole2_velocity + highend_pole_v) / (2.f * highend_pole_v);
//    io_printf(IO_BUF, "rela = %k, angv = %k, maxv = %k\n",
//    		(accum)relative_angular_velocity, (accum)pole_velocity, (accum)highend_pole_v);
    for (int i = 0; i < number_of_bins; i = i + 1){
        if (firing_prob(relative_cart, i)){
            spike_cart(i);
        }
        if (firing_prob(relative_cart_velocity, i)){
            spike_cart_v(i);
        }
        if (firing_prob(relative_angle, i)){
            spike_angle(i);
        }
        if (firing_prob(relative_angular_velocity, i)){
            spike_angle_v(i);
        }
        if (firing_prob(relative_angle_2, i)){
            spike_angle_2(i);
        }
        if (firing_prob(relative_angular_velocity_2, i)){
            spike_angle_2_v(i);
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
    	// Finalise recording
        recording_finalise();

        // go into pause and resume state to avoid another tick
        simulation_handle_pause_resume(resume_callback);

        io_printf(IO_BUF, "infinite_run %d; time %d\n", infinite_run, _time);
        io_printf(IO_BUF, "simulation_ticks %d\n", simulation_ticks);
        //    io_printf(IO_BUF, "key count Left %u\n", left_key_count);
        //    io_printf(IO_BUF, "key count Right %u\n", right_key_count);

        io_printf(IO_BUF, "Exiting on timer.\n");
        simulation_ready_to_read();

        _time -= 1;
        return;
    }
    // Otherwise
    else
    {
        if (_time == 0) {
            update_state(0);
            // possibly use this to allow updating of time whenever
//            auto start = chrono::steady_clock::now();
        }
        // Increment ticks in frame counter and if this has reached frame delay
        tick_in_frame++;
        if (tick_in_frame == time_increment) {
            if (in_bounds){
                max_balance_time = (float)_time;
//                max_balance_time = max_balance_time + 1;
                in_bounds = update_state((float)time_increment / 1000.f);
            }
            // Reset ticks in frame and update frame
            tick_in_frame = 0;
//            update_frame();
            // Update recorded score every 0.1s
            if (score_change_count >= 100) {
                current_state[0] = cart_position;
                current_state[1] = pole_angle;
                current_state[2] = pole2_angle;
                io_printf(IO_BUF, "values: %k %k %k \n", (accum) cart_position,
                		(accum) pole_angle, (accum) pole2_angle);
                if (reward_based == 0) {
                    recording_record(0, &current_state, 12);
                }
                else {
                    recording_record(0, &max_balance_time, 4);
                }
                score_change_count=0;
            }
        }
        if (in_bounds) {
            send_status();
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
	if (!initialize(&timer_period))
	{
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
	io_printf(IO_BUF, "timer tick %d, %k\n", TIMER_TICK, (accum)TIMER_TICK);

	// Register callback
	spin1_callback_on(TIMER_TICK, timer_callback, 2);
    spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);
    spin1_callback_on(MCPL_PACKET_RECEIVED, mc_packet_received_callback, -1);

	_time = UINT32_MAX;

	simulation_run();

}
