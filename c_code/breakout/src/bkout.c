/*
 * Copyright (c) 2013-2019 The University of Manchester
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
#include <math.h>

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
// Game dimension constants
#define GAME_WIDTH_MAX  160
#define GAME_HEIGHT_MAX 128

#define PRINT_GAME_EVOLUTION true

#define NUMBER_OF_LIVES 5
#define SCORE_DOWN_EVENTS_PER_DEATH 5

#define BRICKS_PER_ROW  5
#define BRICKS_PER_COLUMN  2

#define MAX_BALL_SPEED 2

// Ball outof play time (frames)
#define OUT_OF_PLAY 20

// Frame delay (ms)
#define FRAME_DELAY 20

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
typedef enum {
  REGION_SYSTEM,
  REGION_BREAKOUT,
  REGION_RECORDING,
  REGION_PARAM,
} region_t;

typedef enum {
  COLOUR_BACKGROUND = 0x0,
  COLOUR_BAT        = 0x2,
  COLOUR_BALL       = 0x1,
  COLOUR_SCORE      = 0x1,
  COLOUR_BRICK_ON   = 0x1,
  COLOUR_BRICK_OFF  = 0x0
} colour_t;

typedef enum {
  KEY_LEFT  = 0x0,
  KEY_RIGHT = 0x1,
} key_t;

typedef enum {
  SPECIAL_EVENT_SCORE_UP,
  SPECIAL_EVENT_SCORE_DOWN,
  SPECIAL_EVENT_MAX,
} special_event_t;

typedef enum callback_priorities {
    MC = -1, DMA = 0, USER = 0, SDP = 1, TIMER = 2
} callback_priorities;

//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------

//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

static uint32_t _time = 0;
uint32_t pkt_count;

int GAME_WIDTH = 160; // baseline before subsampling
int GAME_HEIGHT = 128; // baseline before subsampling
int y_bits = 8; // baseline to enable building of correct key

// ball coordinates in fixed-point
static int x; //= (GAME_WIDTH / 4) * FACT;
static int y; //= (GAME_HEIGHT - GAME_HEIGHT /8) * FACT;

static int current_number_of_bricks;

static bool bricks[BRICKS_PER_COLUMN][BRICKS_PER_ROW];
bool print_bricks  = true;

int brick_corner_x=-1, brick_corner_y=-1;
int number_of_lives = NUMBER_OF_LIVES;

int x_factor = 1;
int y_factor = 1;
int bricking = 2;

// ball position and velocity scale factor
int FACT = 16;

// ball velocity
int u = MAX_BALL_SPEED;// * FACT;
int v = -MAX_BALL_SPEED;// * FACT;

// bat LHS x position
int x_bat = 32;

// bat length in pixels
int bat_len = 32;

int BRICK_WIDTH = 10;
int BRICK_HEIGHT = 5;

int BRICK_LAYER_OFFSET = 16;
int BRICK_LAYER_HEIGHT = 12;

mars_kiss64_seed_t kiss_seed;

// frame buffer: 160 x 128 x 4 bits: [hard/soft, R, G, B]
static int frame_buff[GAME_WIDTH_MAX/2][GAME_HEIGHT_MAX];

// control pause when ball out of play
static int out_of_play = 0;

// state of left/right keys
static int keystate = 0;

//! The upper bits of the key value that model should transmit with
static uint32_t key;


//! the number of timer ticks that this model should run for before exiting.
uint32_t simulation_ticks = 0;

uint32_t left_key_count = 0;
uint32_t right_key_count = 0;
uint32_t move_count_r = 0;
uint32_t move_count_l = 0;
uint32_t score_change_count=0;
int32_t current_score = 0;

//ratio used in randomising initial x coordinate
static uint32_t x_ratio = UINT32_MAX / GAME_WIDTH_MAX;


//----------------------------------------------------------------------------
// Inline functions
//----------------------------------------------------------------------------
static inline void add_score_up_event(void)
{
  spin1_send_mc_packet(key | (SPECIAL_EVENT_SCORE_UP), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "Score up\n");
  current_score++;
}

static inline void add_score_down_event(void)
{
  spin1_send_mc_packet(key | (SPECIAL_EVENT_SCORE_DOWN), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "Score down\n");
  current_score--;
}

// send packet containing pixel colour change
void add_event(int i, int j, colour_t col, bool bricked)
{
    const uint32_t colour_bit = (col == COLOUR_BACKGROUND) ? 0 : 1;

    const uint32_t spike_key = key | (
        SPECIAL_EVENT_MAX + (i << (y_bits + 2)) + (j << 2) + (bricked<<1) + colour_bit);
//    const uint32_t spike_key = key | (
//    		SPECIAL_EVENT_MAX + (i << (y_bits + colour_bit)) + (j << colour_bit) + colour_bit);

    spin1_send_mc_packet(spike_key, 0, NO_PAYLOAD);
}

// gets pixel colour from within word
static inline colour_t get_pixel_col (int i, int j)
{
  return (colour_t)(frame_buff[i][j]);
}

// inserts pixel colour within word
static inline void set_pixel_col (int i, int j, colour_t col, bool bricked)
{
//    io_printf(IO_BUF, "setting (%d,%d) to %d, b-%d, g%d, u%d, v%d\n",
//    		i, j, col, bricked, y_bits, u, v);
    if (bricked) {
        add_event((brick_corner_x * BRICK_WIDTH),
                      (brick_corner_y* BRICK_HEIGHT + BRICK_LAYER_OFFSET),
                      COLOUR_BACKGROUND, bricked);
    }
//    else if (col != get_pixel_col(i, j))
//    {
    frame_buff[i][j] = col;
    add_event (i, j, col, bricked);
//    }
}

static inline bool is_a_brick(int the_x, int the_y) // x = width, y = height
{
    // io_printf(IO_BUF, "x:%d, y:%d, px:%d, py:%d\n", x, y, pos_x, pos_y);
    if (the_x < 0 || the_y < 0 || the_x >= GAME_WIDTH - 1 || the_y >= GAME_HEIGHT - 1) {
        return false;
    }
    int pos_x=0, pos_y=0;

    if (the_y >= BRICK_LAYER_OFFSET && the_y < BRICK_LAYER_OFFSET + BRICK_LAYER_HEIGHT) {
//        io_printf(IO_BUF, "in brick layer x:%d y:%d px:%d py:%d\n",
//        		the_x, the_y, pos_x, pos_y);
        pos_x = the_x / BRICK_WIDTH;
        pos_y = (the_y - BRICK_LAYER_OFFSET) / BRICK_HEIGHT;
//        io_printf(IO_BUF, "2 in brick layer x:%d y:%d px:%d py:%d\n",
//        		the_x, the_y, pos_x, pos_y);
        bool val = bricks[pos_y][pos_x];
        if (val) {
            add_event(pos_x * BRICK_WIDTH, (pos_y * BRICK_HEIGHT) + BRICK_LAYER_OFFSET,
                COLOUR_BACKGROUND, true);
        }
//        io_printf(IO_BUF, "3 in brick layer x:%d y:%d px:%d py:%d\n",
//        		the_x, the_y, pos_x, pos_y);
        bricks[pos_y][pos_x] = false;
        if (val) {
            brick_corner_x = pos_x;
            brick_corner_y = pos_y;
            current_number_of_bricks--;
        }
        else {
            brick_corner_x = -1;
            brick_corner_y = -1;
        }

//        io_printf(IO_BUF, "x:%d y:%d px:%d py:%d, v:%d\n",
//        		the_x, the_y, pos_x, pos_y, val);
        return val;
    }
    brick_corner_x = -1;
    brick_corner_y = -1;
    return false;
}

static inline uint32_t hitting_a_brick(int the_x, int the_y)
{
    // io_printf(IO_BUF, "hitting brick from x:%d, y:%d\n", the_x, the_y);
    uint32_t encoded_result = 0;
    bool check_left = false;
    bool check_right = false;
    bool check_up = false;
    bool check_down = false;
    if (u > 0) {
        check_right = is_a_brick(the_x + 1, the_y);
        if (check_right) {
            encoded_result = encoded_result + 1;
        }
        // io_printf(IO_BUF, "b1\n");
    }
    else {
        check_left = is_a_brick(the_x - 1, the_y);
        if (check_left) {
            encoded_result = encoded_result + 2;
        }
        // io_printf(IO_BUF, "b2\n");
    }
    if (v > 0){
        check_down = is_a_brick(the_x, the_y + 1);
        if (check_down) {
            encoded_result = encoded_result + 4;
        }
        // io_printf(IO_BUF, "b3\n");
    }
    else {
        check_up = is_a_brick(the_x, the_y - 1);
        if (check_up) {
            encoded_result = encoded_result + 8;
        }
        // io_printf(IO_BUF, "b4\n");
    }

    return encoded_result;
}

//----------------------------------------------------------------------------
// Static functions
//----------------------------------------------------------------------------
// initialise frame buffer to blue
static void init_frame(void)
{
    for (int i=0; i<GAME_WIDTH/4; i++) {
        for (int j=0; j<GAME_HEIGHT; j++) {
            frame_buff[i/4][j] = 0x11111111 * COLOUR_BACKGROUND;
        }
    }

    for (int i =0; i<BRICKS_PER_COLUMN; i++) {
        for (int j=0; j<BRICKS_PER_ROW; j++) {
            if (bricking == 1) {
                bricks[i][j] = true;
            }
            else {
                bricks[i][j] = false;
            }
        }
    }
    current_number_of_bricks = BRICKS_PER_COLUMN * BRICKS_PER_ROW;
}

float rand021(void)
{
    return (float)(mars_kiss64_seed(kiss_seed) / (float)0xffffffff);
}

static void update_frame (uint32_t time)
{
  if (PRINT_GAME_EVOLUTION) {
    io_printf(IO_BUF, "time = %u, t20xf = %u\n", time, time % (20 * x_factor));
  }

    // Draw bat
    // Cache old bat position
    const int old_xbat = x_bat;
    int move_direction;
    if (right_key_count > left_key_count) {
        move_direction = KEY_RIGHT;
        move_count_r++;
    //    io_printf(IO_BUF, "moved right\n");
    }
    else if (left_key_count > right_key_count) {
        move_direction = KEY_LEFT;
        move_count_l++;
        //    io_printf(IO_BUF, "moved left\n");
    }
    else {
        move_direction = 2;
        //    io_printf(IO_BUF, "didn't move!\n");
    }

    // Update bat and clamp
    if (move_direction == KEY_LEFT && --x_bat < 0) {
        x_bat = 1;
    }
    else if (move_direction == KEY_RIGHT && ++x_bat > GAME_WIDTH-bat_len) {
        x_bat = GAME_WIDTH - bat_len - 1;
    }

    // Clear keystate
    left_key_count = 0;
    right_key_count = 0;

    // If bat's moved
    if (old_xbat != x_bat) {
        // Draw bat pixels
        // io_printf(IO_BUF, "oxb:%d, xb:%d, bl:%d\n", old_xbat, x_bat, bat_len);
        for (int i = x_bat; i < (x_bat + bat_len); i++) {
            set_pixel_col(i, GAME_HEIGHT-1, COLOUR_BAT, false);
        }
        // Remove pixels left over from old bat
        if (x_bat > old_xbat) {
            set_pixel_col(old_xbat, GAME_HEIGHT-1, COLOUR_BACKGROUND, false);
        }
        else if (x_bat < old_xbat) {
            set_pixel_col(old_xbat + bat_len - 1, GAME_HEIGHT-1, COLOUR_BACKGROUND, false);
        }

        //only draw left edge of bat pixel
        // add_event(x_bat, GAME_HEIGHT-1, COLOUR_BAT);
        //send off pixel to network (ignoring game frame buffer update)
        // add_event (old_xbat, GAME_HEIGHT-1, COLOUR_BACKGROUND);
    }

    // draw ball
    if (out_of_play == 0) {
         if (time % (20 * x_factor) == 0) {
            // clear pixel to background
          if (PRINT_GAME_EVOLUTION) {
            io_printf(IO_BUF, "setting ball to background x=%d, y=%d, u=%d, v=%d\n",
                x, y, u, v);
          }

            if (get_pixel_col(x, y) != COLOUR_BAT) {
                set_pixel_col(x, y, COLOUR_BACKGROUND, false);
            }

            // move ball in x and bounce off sides
            x += u;
            if (x + u < 0) {
                // io_printf(IO_BUF, "OUT 1\n");
                u = -u;
            }
            if (x + u >= GAME_WIDTH) {
//                io_printf(IO_BUF, "OUT 2 x = %d, u = %d, gw = %d, fact = %d\n",
//                		x, u, GAME_WIDTH, FACT);
                u = -u;
            }

            // move ball in y and bounce off top
            y += v;

            if (y + v > GAME_HEIGHT) {
                y = GAME_HEIGHT - 1;
            }
            if (y + v < 0) {
                v = -v;
            }

            uint32_t encoded_result = hitting_a_brick(x, y);
            bool brick_direction[4];// = hitting_a_brick(x+(u/2), y+(v/2));
            for (int i = 0; i < 4; i++) {
                if (((encoded_result >> i) & 1 ) == 1) {
                    brick_direction[i] = true;
                }
                else {
                    brick_direction[i] = false;
                }
            }
//            io_printf(IO_BUF, "b1:%d, b2:%d, b3:%d, b4:%d, u:%d, v:%d, result:%u\n",
//            		brick_direction[0], brick_direction[1], brick_direction[2],
//					brick_direction[3], u, v, encoded_result);
            if (brick_direction[0] || brick_direction[1] || brick_direction[2] ||
                brick_direction[3]) {
                set_pixel_col(x, y, COLOUR_BACKGROUND, false);

                if (brick_direction[0]) {
                    u = -u;
                    brick_direction[0] = false;
                    add_score_up_event();
                    // io_printf(IO_BUF, "ob1\n");
                }
                if (brick_direction[1]) {
                    u = -u;
                    brick_direction[1] = false;
                    add_score_up_event();
                    // io_printf(IO_BUF, "ob2\n");
                }
                if (brick_direction[2]) {
                    v = -v;
                    brick_direction[2] = false;
                    add_score_up_event();
                    // io_printf(IO_BUF, "ob3\n");
                }
                if (brick_direction[3]) {
                    v = -v;
                    brick_direction[3] = false;
                    add_score_up_event();
                    // io_printf(IO_BUF, "ob4\n");
                }
            }
            else {
                uint32_t encoded_result = hitting_a_brick(x+(u/2), y+(v/2));
                bool brick_direction[4];// = hitting_a_brick(x+(u/2), y+(v/2));
                for (int i = 0; i < 4; i++){
                    if (((encoded_result >> i) & 1) == 1) {
                        brick_direction[i] = true;
                    }
                    else {
                        brick_direction[i] = false;
                    }
                }
                // bool * brick_direction = hitting_a_brick(x, y);
//                io_printf(IO_BUF, "2 b1:%d, b2:%d, b3:%d, b4:%d\n", brick_direction[0],
//                		brick_direction[1], brick_direction[2], brick_direction[3]);
                if (brick_direction[0] || brick_direction[1] || brick_direction[2] ||
                    brick_direction[3]) {
                    x = x + (u / 2);
                    y = y + (v / 2);
                    if (brick_direction[0]) {
                        u = -u;
                        brick_direction[0] = false;
                        add_score_up_event();
                    }
                    if (brick_direction[1]) {
                        u = -u;
                        brick_direction[1] = false;
                        add_score_up_event();
                    }
                    if (brick_direction[2]) {
                        v = -v;
                        brick_direction[2] = false;
                        add_score_up_event();
                    }
                    if (brick_direction[3]) {
                        v = -v;
                        brick_direction[3] = false;
                        add_score_up_event();
                    }
                }
            }

            if (get_pixel_col(x, y) == COLOUR_BAT || get_pixel_col(
                x + (u / 2), y + (v / 2)) == COLOUR_BAT || get_pixel_col(
                    x, y + (v / 2)) == COLOUR_BAT) {
              if (PRINT_GAME_EVOLUTION) {
                io_printf(IO_BUF, "got in hitting bat x=%d, y=%d, u=%d, v=%d\n",
                    x, y, u, v);
              }
                if (x < (x_bat + bat_len/4)) {
                  if (PRINT_GAME_EVOLUTION){
                    io_printf(IO_BUF, "BAT 1");
                  }
                    u = -MAX_BALL_SPEED;
                    v = -v;
                }
                else if (x < (x_bat + (bat_len/2))) {
                  if (PRINT_GAME_EVOLUTION) {
                    io_printf(IO_BUF, "BAT 2");
                  }
                    u = -(MAX_BALL_SPEED / 2);
                    v = -v;
                }
                else if (x < (x_bat + ((3 * bat_len) / 4))) {
                  if (PRINT_GAME_EVOLUTION) {
                    io_printf(IO_BUF, "BAT 3");
                  }
                    u = (MAX_BALL_SPEED / 2);
                    v = -v;
                }
                else if (x < (x_bat + bat_len)) {
                  if (PRINT_GAME_EVOLUTION) {
                    io_printf(IO_BUF, "BAT 4");
                  }
                    u = MAX_BALL_SPEED;
                    v = -v;
                }
                else {
                  if (PRINT_GAME_EVOLUTION) {
                    io_printf(IO_BUF, "Broke bat 0x%x\n", frame_buff[x][y]);
                  }
                }

                // Increase score
                if (!bricking){
                    add_score_up_event();
                }
            }

            // lost ball
            if (y + v > GAME_HEIGHT) {
              if (PRINT_GAME_EVOLUTION) {
                io_printf(IO_BUF, "got in lost ball x=%d, y=%d, u=%d, v=%d\n",
                    x, y, u, v);
              }
                v = -MAX_BALL_SPEED;
                // todo make this random in some respect or non abusable
                x = x_bat + (bat_len / 2);
                y = GAME_HEIGHT - 2;

                if (mars_kiss64_seed(kiss_seed) > 0x7FFFFFFF) {
                    //        io_printf(IO_BUF, "MARS 1");
                    u = -MAX_BALL_SPEED;
                }
                else {
                    u = MAX_BALL_SPEED;
                }
                //      x = (int)(mars_kiss32()%GAME_WIDTH);
                //      io_printf(IO_BUF, "random x = %d", x);

                out_of_play = OUT_OF_PLAY;
                // Decrease score
                number_of_lives--;
//                if (!number_of_lives && bricking){
//                    for (int i=0; i<SCORE_DOWN_EVENTS_PER_DEATH; i++) {
//                        add_score_down_event();
//                    }
//                    number_of_lives = NUMBER_OF_LIVES;
//                }
//                else {
                    add_score_down_event();
//                }
                if (PRINT_GAME_EVOLUTION) {
                  io_printf(IO_BUF, "after reset x=%d, y=%d, u=%d, v=%d\n",
                      x, y, u, v);
                }
            }
            // draw ball
            else
            {
//                io_printf(IO_BUF, "else x=%d, y=%d, u=%d, v=%d\n", x, y, u, v);
                if (get_pixel_col(x, y) != COLOUR_BAT){
                    set_pixel_col(x, y, COLOUR_BALL, false);
                }
            }
        }
    }
    else
    {
        --out_of_play;
    }
}

static bool initialize(uint32_t *timer_period)
{
    io_printf(IO_BUF, "Initialise breakout: started\n");

    // Get the address this core's DTCM data starts at from SRAM
    data_specification_metadata_t *ds_regions = data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(ds_regions)) {
        return false;
    }

    // Get the timing details and set up thse simulation interface
    if (!simulation_initialise(data_specification_get_region(REGION_SYSTEM, ds_regions),
        APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
      &infinite_run, &_time, SDP, DMA)) {
        return false;
    }

    io_printf(IO_BUF, "simulation time = %u\n", simulation_ticks);
    io_printf(IO_BUF, "\tTimer period=%d\n", *timer_period);

    // Read breakout region memory
    address_t breakout_region = data_specification_get_region(REGION_BREAKOUT, ds_regions);

    key = breakout_region[0];
    io_printf(IO_BUF, "\tKey=%08x\n", key);

    //get recording region
    void *recording_region = data_specification_get_region(
        REGION_RECORDING, ds_regions);

    // Read param region to initialise game parameters
    address_t param_region = data_specification_get_region(REGION_PARAM, ds_regions);

    x_factor = param_region[0];
    y_factor = param_region[1];
    bricking = param_region[2];
    kiss_seed[0] = param_region[3];
    kiss_seed[1] = param_region[4];
    kiss_seed[2] = param_region[5];
    kiss_seed[3] = param_region[6];

    io_printf(IO_BUF, "x_factor = %d, y_factor = %d, bricking = %d, seed = [%d, %d, %d, %d]\n",
            x_factor, y_factor, bricking, kiss_seed[0], kiss_seed[1], kiss_seed[2], kiss_seed[3]);

    if (bricking != 0 && bricking != 1) {
        io_printf(IO_BUF, "\n Brick setting invalid at: %d \n", bricking);
        return false;
    }

    // Setup game environment
    GAME_WIDTH = GAME_WIDTH / x_factor;
    GAME_HEIGHT = GAME_HEIGHT / y_factor;

    io_printf(IO_BUF, "game w = %d, game h = %d, x=%d, y=%d, u=%d, v=%d, xf=%d, yf=%d\n",
      GAME_WIDTH, GAME_HEIGHT, x, y, u, v, x_factor, y_factor);

    x_bat = x_bat / x_factor;

    bat_len = bat_len / x_factor;

    x = x_bat + (bat_len / 2);
    y = GAME_HEIGHT - 2;

//    frame_buff[GAME_WIDTH / 8][GAME_HEIGHT];

    x_ratio = UINT32_MAX / GAME_WIDTH;

    // rescale variables
//    FACT = FACT / y_factor;

    v = -MAX_BALL_SPEED;
    if (rand021() < 0.5){
        u = MAX_BALL_SPEED;
    }
    else{
        u = -MAX_BALL_SPEED;
    }

    BRICK_WIDTH = GAME_WIDTH / BRICKS_PER_ROW;//BRICK_WIDTH / x_factor;
    BRICK_HEIGHT = 16 / y_factor;//BRICK_HEIGHT / y_factor;

    BRICK_LAYER_OFFSET = BRICK_LAYER_OFFSET / y_factor;
    BRICK_LAYER_HEIGHT = BRICKS_PER_COLUMN * BRICK_HEIGHT;//BRICK_LAYER_HEIGHT / y_factor;

    y_bits = ceil(log2(GAME_HEIGHT));

    io_printf(IO_BUF, "x:%d, y:%d, b_width:%d, brick_height:%d, b_lay_off:%d, b_lay_hei:%d\n"
        "x_bat:%d, bat_len:%d, u:%d, v:%d, y_bits:%d\n",
      x, y, BRICK_WIDTH, BRICK_HEIGHT, BRICK_LAYER_OFFSET, BRICK_LAYER_HEIGHT,
      x_bat, bat_len, u, v, y_bits);

    // Setup recording
    uint32_t recording_flags = 0;
    if (!recording_initialize(&recording_region, &recording_flags)) {
        rt_error(RTE_SWERR);
        return false;
    }

    io_printf(IO_BUF, "Initialise: completed successfully\n");

    return true;
}

void resume_callback(void)
{
    recording_reset();
}

void timer_callback(uint unused, uint dummy)
{
//    io_printf(IO_BUF, "time = %d", _time);
    use(unused);
    use(dummy);

    _time++;
    score_change_count++;

    if (!infinite_run && _time >= simulation_ticks) {
        io_printf(IO_BUF, "time = %d\n", _time);

        // Finalise recording
        recording_finalise();
        io_printf(IO_BUF, "done recording\n");

        // go into pause and resume state to avoid another tick
        simulation_handle_pause_resume(resume_callback);

        io_printf(IO_BUF, "move count Left %u\n", move_count_l);
        io_printf(IO_BUF, "move count Right %u\n", move_count_r);
        io_printf(IO_BUF, "infinite_run %d; time %d\n", infinite_run, _time);
        io_printf(IO_BUF, "simulation_ticks %d\n", simulation_ticks);
        // io_printf(IO_BUF, "key count Left %u", left_key_count);
        // io_printf(IO_BUF, "key count Right %u", right_key_count);

        io_printf(IO_BUF, "Exiting on timer.\n");
        simulation_ready_to_read();

        _time -= 1;
        return;
    }
    // Otherwise
    else
    {
        // Increment ticks in frame counter and if this has reached frame delay
        // io_printf(IO_BUF, "else time = %d\n", _time);
        if (_time % 20 == 0) {
            if (!current_number_of_bricks && bricking == 1) {
                for (int i =0; i<BRICKS_PER_COLUMN; i++) {
                    for (int j=0; j<BRICKS_PER_ROW; j++) {
                        bricks[i][j] = true;
                    }
                }
                current_number_of_bricks = BRICKS_PER_COLUMN * BRICKS_PER_ROW;
                set_pixel_col(x, y, COLOUR_BACKGROUND, false);
                // print_bricks = true;
                v = -MAX_BALL_SPEED;
                // todo make this random in some respect or not
                y = GAME_HEIGHT - 2;

                if (mars_kiss64_seed(kiss_seed) > 0x7FFFFFFF) {
                    // io_printf(IO_BUF, "MARS 2");
                    u = -u;
                }

                //randomises initial x location
                x = x_bat + (bat_len / 2);
            }

//            if (print_bricks) {
//            	print_bricks = false;
//            }
            for (int i =0; i<BRICKS_PER_COLUMN; i++) {
                for (int j=0; j<BRICKS_PER_ROW; j++) {
                    if (bricks[i][j]) {
                        // io_printf(IO_BUF, "adding brick event at i:%d j:%d\n", i, j);
                        add_event(j * BRICK_WIDTH, (i * BRICK_HEIGHT) + BRICK_LAYER_OFFSET,
                            COLOUR_BRICK_ON, true);
                    }
                }
            }
            // If this is the first update, draw bat as
            // collision detection relies on this
            if (_time == FRAME_DELAY) {
//                io_printf(IO_BUF, "sets the bat for the first time bl:%d, xb:%, gh:%d\n",
//                		bat_len, x_bat, GAME_HEIGHT);
                // Draw bat
                for (int i = x_bat; i < (x_bat + bat_len); i++)
                {
                    set_pixel_col(i, GAME_HEIGHT-1, COLOUR_BAT, false);
                }
            }
            update_frame(_time);
            // Update recorded score every 1s
            if (score_change_count>=1000) {
                bool record_check = recording_record(0, &current_score, 4);
                io_printf(IO_BUF, "record outcome %d when recording %d\n",
                    record_check, current_score);
                score_change_count=0;
            }
        }
    }
}

void mc_packet_received_callback(uint key, uint payload)
{
    // If no payload has been set, make sure the loop will run
    if (payload == 0) { payload = 1; }

    for (uint count = payload; count > 0; count--) {
        // Right
        if (key & KEY_RIGHT) {
            right_key_count++;
        }
        // Left
        else {
            left_key_count++;
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
        io_printf(IO_BUF, "Init error!\n");
        rt_error(RTE_SWERR);
        return;
    }

    init_frame();
    keystate = 0; // IDLE
    pkt_count = 0;

    // Set timer tick (in microseconds)
    io_printf(IO_BUF, "setting timer tick callback for %d microseconds\n",
              timer_period);
    spin1_set_timer_tick(timer_period);
    io_printf(IO_BUF, "bricks %x\n", &bricks);

    io_printf(IO_BUF, "simulation_ticks %d\n", simulation_ticks);

    // Register callback
    spin1_callback_on(TIMER_TICK, timer_callback, TIMER);
    spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, MC);
    spin1_callback_on(MCPL_PACKET_RECEIVED, mc_packet_received_callback, MC);

    _time = UINT32_MAX;

    simulation_run();
}
