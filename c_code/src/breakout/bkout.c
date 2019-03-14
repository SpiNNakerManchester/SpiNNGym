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

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
//typedef enum
//{
//  REGION_SYSTEM,
//  REGION_BREAKOUT,
//  REGION_RECORDING,
//  REGION_PARAM,
//} region_t;

// Read param region
//address_t address = data_specification_get_data_address();
//address_t param_region = data_specification_get_region(REGION_PARAM, address);
//GAME_WIDTH_MAX = param_region[0]
//GAME_HEIGHT_MAX = param_region[1]

#define NUMBER_OF_LIVES 5
#define SCORE_DOWN_EVENTS_PER_DEATH 5

#define MAX_BRICKS_PER_ROW  32
#define MAX_BRICKS_PER_COLUMN  2
int BRICKS_PER_ROW = 32;
int BRICKS_PER_COLUMN = 2;

#define MAX_BALL_SPEED 2

// Ball outof play time (frames)
#define OUT_OF_PLAY 100

// Frame delay (ms)
#define FRAME_DELAY 20 //14//20

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
typedef enum
{
  REGION_SYSTEM,
  REGION_BREAKOUT,
  REGION_RECORDING,
  REGION_PARAM,
} region_t;

typedef enum
{
//  COLOUR_HARD       = 0x8,
//  COLOUR_SOFT       = 0x0,
//  COLOUR_BRICK      = 0x10,
//
//  COLOUR_BACKGROUND = COLOUR_SOFT | 0x1,
//  COLOUR_BAT        = COLOUR_HARD | 0x6,
//  COLOUR_BALL       = COLOUR_HARD | 0x7,
//  COLOUR_SCORE      = COLOUR_SOFT | 0x6,
//  COLOUR_BRICK_ON   = COLOUR_BRICK | 0x0,
//  COLOUR_BRICK_OFF  = COLOUR_BRICK | 0x1

  COLOUR_BACKGROUND = 0x0,
  COLOUR_BAT        = 0x2,
  COLOUR_BALL       = 0x1,
  COLOUR_SCORE      = 0x1,
  COLOUR_BRICK_ON   = 0x1,
  COLOUR_BRICK_OFF  = 0x0
} colour_t;

typedef enum
{
  KEY_LEFT  = 0x0,
  KEY_RIGHT = 0x1,
} key_t;

typedef enum
{
  SPECIAL_EVENT_SCORE_UP,
  SPECIAL_EVENT_SCORE_DOWN,
  SPECIAL_EVENT_MAX,
} special_event_t;

//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------


//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

static uint32_t _time = 0;
uint32_t pkt_count;

int GAME_WIDTH = 160;
int GAME_HEIGHT = 128;
int y_bits = 8;

// initial ball coordinates in fixed-point
static int x; //= (GAME_WIDTH / 4) * FACT;
static int y; //= (GAME_HEIGHT - GAME_HEIGHT /8) * FACT;

static int current_number_of_bricks;

static bool bricks[MAX_BRICKS_PER_COLUMN][MAX_BRICKS_PER_ROW];
bool print_bricks  = true;

int brick_corner_x=-1, brick_corner_y=-1;
int number_of_lives = NUMBER_OF_LIVES;

int x_factor = 1;
int y_factor = 1;
int bricking = 2;

// ball position and velocity scale factor
int FACT = 16;

// initial ball velocity in fixed-point
int u = MAX_BALL_SPEED;// * FACT;
int v = -MAX_BALL_SPEED;// * FACT;

// bat LHS x position
int x_bat = 32;

// bat length in pixels
int bat_len = 48;

int BRICK_WIDTH = 10;
int BRICK_HEIGHT = 6;

int BRICK_LAYER_OFFSET = 16;
int BRICK_LAYER_HEIGHT = 12;

mars_kiss64_seed_t kiss_seed;

// frame buffer: 160 x 128 x 4 bits: [hard/soft, R, G, B]
static int frame_buff[GAME_WIDTH_MAX/4][GAME_HEIGHT_MAX];

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
static inline void add_score_up_event()
{
  spin1_send_mc_packet(key | (SPECIAL_EVENT_SCORE_UP), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "Score up\n");
  current_score++;
}

static inline void add_score_down_event()
{
  spin1_send_mc_packet(key | (SPECIAL_EVENT_SCORE_DOWN), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "Score down\n");
  current_score--;
}

// send packet containing pixel colour change
void add_event(int i, int j, colour_t col, bool bricked)
{
    const uint32_t colour_bit = (col == COLOUR_BACKGROUND) ? 0 : 1;
    io_printf(IO_BUF, "sending payload back i:%d, j:%d, c:%d, b:%d\n", i, j, colour_bit, bricked);

//    int row_bits = (np.ceil(np.log2(x_res)));
//    int idx = 0;
//
//    if (colour_bit):
//        idx = 1;
//
//    int row += 0
//    idx = idx | (row << (colour_bits));  // colour bit
//    idx = idx | (col << (row_bits + colour_bits));
//
//    // add two to allow for special event bits
//    idx = idx + 2;
//    io_printf(IO_BUF, "sending fixed payload back i:%d, j:%d, c:%d, b:%d\n", i, j, colour_bit, bricked);
//    io_printf(IO_BUF, "fixed payload: full:%d, nokey:%d, bits:%d\n",
//        key | (SPECIAL_EVENT_MAX + (i << (y_bits + 2)) + (j << 2) + (bricked<<1) + colour_bit),
//        (SPECIAL_EVENT_MAX + (i << (y_bits + 2)) + (j << 2) + (bricked<<1) + colour_bit),
//        (i << (y_bits + 2)) + (j << 2) + (bricked<<1) + colour_bit);


    const uint32_t spike_key = key | (SPECIAL_EVENT_MAX + (i << (y_bits + 2)) + (j << 2) + (bricked<<1) + colour_bit);
//    const uint32_t spike_key = key | (SPECIAL_EVENT_MAX + (i << (y_bits + colour_bit)) + (j << colour_bit) + colour_bit);

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
//    io_printf(IO_BUF, "setting (%d,%d) to %d, b-%d, g%d, u%d, v%d\n", i, j, col, bricked, y_bits, u, v);
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

static inline bool is_a_brick(int the_x, int the_y) // x - width, y- height?
{

//    io_printf(IO_BUF, "%d %d %d %d\n", x, y, pos_x, pos_y);
    if (the_x < 0 || the_y < 0 || the_x >= GAME_WIDTH - 1 || the_y >= GAME_HEIGHT - 1){
        return false;
    }
    int pos_x=0, pos_y=0;

    if ( the_y >= BRICK_LAYER_OFFSET && the_y < BRICK_LAYER_OFFSET + BRICK_LAYER_HEIGHT) {
        pos_x = the_x / BRICK_WIDTH;
        pos_y = (the_y - BRICK_LAYER_OFFSET) / BRICK_HEIGHT;
        bool val = bricks[pos_y][pos_x];
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

        io_printf(IO_BUF, "x:%d y:%d px:%d py:%d, v:%d\n", the_x, the_y, pos_x, pos_y, val);
        return val;
    }
    brick_corner_x = -1;
    brick_corner_y = -1;
    return false;
}



//----------------------------------------------------------------------------
// Static functions
//----------------------------------------------------------------------------
// initialise frame buffer to blue
static void init_frame ()
{
    for (int i=0; i<GAME_WIDTH/4; i++)
    {
        for (int j=0; j<GAME_HEIGHT; j++)
        {
            frame_buff[i/4][j] = 0x11111111 * COLOUR_BACKGROUND;
        }
    }

    for (int i =0; i<BRICKS_PER_COLUMN; i++)
        for (int j=0; j<BRICKS_PER_ROW; j++) {
            if(bricking == 1){
                bricks[i][j] = true;
            }
            else{
                bricks[i][j] = false;
            }
        }
    current_number_of_bricks = BRICKS_PER_COLUMN * BRICKS_PER_ROW;
}

float rand021(){
    return (float)(mars_kiss64_seed(kiss_seed) / (float)0xffffffff);
}

static void update_frame (uint32_t time)
{
//    io_printf(IO_BUF, "time = %u, t20xf = %u\n", time, time % (20 * x_factor));
    // draw bat
    // Cache old bat position
    const uint32_t old_xbat = x_bat;
    int move_direction;
    if (right_key_count > left_key_count){
        move_direction = KEY_RIGHT;
        move_count_r++;
    //    io_printf(IO_BUF, "moved right\n");
    }
    else if (left_key_count > right_key_count){
        move_direction = KEY_LEFT;
        move_count_l++;
        //    io_printf(IO_BUF, "moved left\n");
    }
    else{
        move_direction = 2;
        //    io_printf(IO_BUF, "didn't move!\n");
    }
//    io_printf(IO_BUF, "left = %d, right = %d\n", left_key_count, right_key_count);


    // Update bat and clamp
    if (move_direction == KEY_LEFT && --x_bat < 0)
    {
        x_bat = 0;
    }
    else if (move_direction == KEY_RIGHT && ++x_bat > GAME_WIDTH-bat_len)
    {
        x_bat = GAME_WIDTH-bat_len;
    }

    // Clear keystate
    left_key_count = 0;
    right_key_count = 0;

    // If bat's moved
    if (old_xbat != x_bat)
    {
        // Draw bat pixels
//        io_printf(IO_BUF, "oxb:%d, xb:%d, bl:%d\n", old_xbat, x_bat, bat_len);
        for (int i = x_bat; i < (x_bat + bat_len); i++)
        {
            set_pixel_col(i, GAME_HEIGHT-1, COLOUR_BAT, false);
        }
        // Remove pixels left over from old bat
        if (x_bat > old_xbat)
        {
            set_pixel_col(old_xbat, GAME_HEIGHT-1, COLOUR_BACKGROUND, false);
        }
        else if (x_bat < old_xbat)
        {
            set_pixel_col(old_xbat + bat_len - 1, GAME_HEIGHT-1, COLOUR_BACKGROUND, false);
        }

        //only draw left edge of bat pixel
        // add_event(x_bat, GAME_HEIGHT-1, COLOUR_BAT);
        //send off pixel to network (ignoring game frame buffer update)
        // add_event (old_xbat, GAME_HEIGHT-1, COLOUR_BACKGROUND);
    }

    // draw ball
    if (out_of_play == 0)
    {
         if (time % (20 * x_factor) == 0)
         {
            // clear pixel to background
    //        io_printf(IO_BUF, "setting ball to background x=%d, y=%d, u=%d, v=%d\n", x, y, u, v);
            set_pixel_col(x, y, COLOUR_BACKGROUND, false);

            // move ball in x and bounce off sides
            x += u;
            if (x + u < 0)
            {
                //      io_printf(IO_BUF, "OUT 1\n");
                u = -u;
            }
            if (x + u >= GAME_WIDTH)
            {
                //      io_printf(IO_BUF, "OUT 2 x = %d, u = %d, gw = %d, fact = %d\n", x, u, GAME_WIDTH, FACT);
                u = -u;
            }

            // move ball in y and bounce off top
            y += v;
            // if ball entering bottom row, keep it out XXX SD
    //        if (y == GAME_HEIGHT - 1)
            if (y + v > GAME_HEIGHT)
            {
                y = GAME_HEIGHT;
            }
            if (y + v < 0)
            {
                v = -v;
            }

    //        io_printf(IO_BUF, "about to is a brick x=%d, y=%d, u=%d, v=%d\n", x, y, u, v);
            //detect collision
            // if we hit something hard! -- paddle or brick
//            bool bricked = is_a_brick(x, y);
//            bool bricked = is_a_brick(x+(u/2), y+(v/2));
            bool bricked1 = false;
            bool bricked2 = false;
            bricked1 = is_a_brick(x+(u/2), y+(v/2));
            if (!bricked1){
                bricked2 = is_a_brick(x, y);
            }
            bool bricked = bricked1 | bricked2;
            if (bricked) {
                int brick_x = brick_corner_x * BRICK_WIDTH;
                int brick_y = (brick_corner_y * BRICK_HEIGHT) + BRICK_LAYER_OFFSET;
                io_printf(IO_BUF, "got in bricked, u:%d, v%d, x:%d, y:%d, brick_x:%d, brick_y:%d, brick_c_x:%d, brick_c_y:%d, b1:%d\n", u, v, x, y, brick_x, brick_y, brick_corner_x, brick_corner_y, bricked1);
                //        io_printf(IO_BUF, "x-brick_x = %d, %d %d\n",x/FACT - brick_x, x/FACT, brick_x);
                //        io_printf(IO_BUF, "y-brick_y = %d, %d %d",y/FACT - brick_y, y/FACT, brick_y);
//                int the_x = x;
//                int the_y = y;
                if (bricked1){
                    set_pixel_col(x, y, COLOUR_BACKGROUND, bricked);
                    int x = x+(u/2);
                    int y = y+(v/2);
                }
                if (!current_number_of_bricks){
                   set_pixel_col(x, y, COLOUR_BACKGROUND, bricked);
                }
//                if (brick_x == the_x && u > 0){
//                    u = -u;
//                }
//                else if (the_x == brick_x + BRICK_WIDTH - 1 && u < 0){
//                    u = -u;
//                }
//                if (brick_y == the_y && v > 0){
//                    v = -v;
//                }
//                else if (the_y ==  brick_y + BRICK_HEIGHT - 1 && v < 0){
//                    v = -v;
//                }
//                if (brick_x + BRICK_WIDTH <= x - u){// && u < 0){
//                    u = -u;
//                }
//                else if (x - u <= brick_x){// && u > 0){
//                    u = -u;
//                }
//                if (brick_y + BRICK_HEIGHT <= y - v){// && v < 0){
//                    v = -v;
//                }
//                else if (y - v <=  brick_y){// && v > 0){
//                    v = -v;
//                }
                if (brick_x + BRICK_WIDTH == x){// + (u/2)){// && u < 0){
                    u = -u;
                }
                else if (x == brick_x){// && u > 0){
                    u = -u;
                }
                if (brick_y + BRICK_HEIGHT == y){// + (v/2)){// && v < 0){
                    v = -v;
                }
                else if (y ==  brick_y){// && v > 0){
                    v = -v;
                }

                set_pixel_col(x, y, COLOUR_BACKGROUND, bricked);

                bricked = false;
                // Increase score
                add_score_up_event();
            }

            if (get_pixel_col(x, y+v-(v/2)) == COLOUR_BAT)
            {
                io_printf(IO_BUF, "got in hitting bat x=%d, y=%d, u=%d, v=%d\n", x, y, u, v);
                bool broke = false;
                if (x < (x_bat + bat_len/4))
                {
                    io_printf(IO_BUF, "BAT 1");
                    u = -MAX_BALL_SPEED;
                    v = -v;
                }
                else if (x < (x_bat + (bat_len/2)))
                {
                    io_printf(IO_BUF, "BAT 2");
                    u = -(MAX_BALL_SPEED / 2);
                    v = -v;
                }
                else if (x < (x_bat + ((3 * bat_len) / 4)))
                {
                    io_printf(IO_BUF, "BAT 3");
                    u = (MAX_BALL_SPEED / 2);
                    v = -v;
                }
                else if (x < (x_bat + bat_len))
                {
                    io_printf(IO_BUF, "BAT 4");
                    u = MAX_BALL_SPEED;
                    v = -v;
                }
                else
                {
                    io_printf(IO_BUF, "Broke bat 0x%x\n", frame_buff[x][y]);
                    broke = true;
                }

    //            if (broke == false)
    //            {
    //              v = -MAX_BALL_SPEED / x_factor;
    //              y -= 16 / y_factor;
    //            }
                // Increase score
                if (!bricking){
                    add_score_up_event();
                }
            }

            // lost ball
    //        if (y >= GAME_HEIGHT - v)
            if (y + v > GAME_HEIGHT)
            {
                io_printf(IO_BUF, "got in lost ball x=%d, y=%d, u=%d, v=%d\n", x, y, u, v);
                v = -MAX_BALL_SPEED;
                //todo make this random in some respect or not
                x = x_bat + (bat_len / 2);
                y = GAME_HEIGHT - 2;

                if(mars_kiss64_seed(kiss_seed) > 0x7FFFFFFF){
                    //        io_printf(IO_BUF, "MARS 1");
                    u = -MAX_BALL_SPEED;
                }
                else{
                    u = MAX_BALL_SPEED;
                }
                //      x = (int)(mars_kiss32()%GAME_WIDTH);
                //      io_printf(IO_BUF, "random x = %d", x);

                out_of_play = OUT_OF_PLAY;
                // Decrease score
                number_of_lives--;
                if (!number_of_lives && bricking){
                    for(int i=0; i<SCORE_DOWN_EVENTS_PER_DEATH;i++) {
                        add_score_down_event();
                    }
                    number_of_lives = NUMBER_OF_LIVES;
                }
                else {
                    add_score_down_event();
                }
                io_printf(IO_BUF, "after reset x=%d, y=%d, u=%d, v=%d\n", x, y, u, v);
            }
            // draw ball
            else
            {
    //            io_printf(IO_BUF, "else x=%d, y=%d, u=%d, v=%d\n", x, y, u, v);
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
    address_t address = data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(address))
    {
        return false;
    }
    // Get the timing details and set up thse simulation interface
    if (!simulation_initialise(data_specification_get_region(REGION_SYSTEM, address),
    APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
    &infinite_run, 1, NULL))
    {
        return false;
    }
    io_printf(IO_BUF, "simulation time = %u\n", simulation_ticks);


    // Read breakout region
    address_t breakout_region = data_specification_get_region(REGION_BREAKOUT, address);
    key = breakout_region[0];
    io_printf(IO_BUF, "\tKey=%08x\n", key);
    io_printf(IO_BUF, "\tTimer period=%d\n", *timer_period);

    //get recording region
    address_t recording_address = data_specification_get_region(
                                       REGION_RECORDING,address);

    // Read param region
    address_t param_region = data_specification_get_region(REGION_PARAM, address);

    x_factor = param_region[0];
    y_factor = param_region[1];
    bricking = param_region[2];
    kiss_seed[0] = param_region[3];
    kiss_seed[1] = param_region[4];
    kiss_seed[2] = param_region[5];
    kiss_seed[3] = param_region[6];
    io_printf(IO_BUF, "x_factor = %d, y_factor = %d, bricking = %d, seed = [%d, %d, %d, %d]/[%u, %u, %u, %u]\n",
                x_factor, y_factor, bricking, kiss_seed[0], kiss_seed[1], kiss_seed[2], kiss_seed[3],
                kiss_seed[0], kiss_seed[1], kiss_seed[2], kiss_seed[3]);

    if(bricking != 0 && bricking != 1){
        io_printf(IO_BUF, "\nbricking is broke af\n");
    }

    GAME_WIDTH = GAME_WIDTH / x_factor;

    GAME_HEIGHT = GAME_HEIGHT / y_factor;

    //todo make this random in some respect

    io_printf(IO_BUF, "game w = %d, game h = %d, x=%d, y=%d, u=%d, v=%d, xf=%d, yf=%d\n", GAME_WIDTH, GAME_HEIGHT, x, y, u, v, x_factor, y_factor);

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

    BRICKS_PER_ROW = MAX_BRICKS_PER_ROW / x_factor;
    BRICKS_PER_COLUMN = MAX_BRICKS_PER_COLUMN;
    io_printf(IO_BUF, "BPR = %d, BPC = %d\n", BRICKS_PER_ROW, BRICKS_PER_COLUMN);

    BRICK_WIDTH = GAME_WIDTH / BRICKS_PER_ROW;//BRICK_WIDTH / x_factor;
    BRICK_HEIGHT = 16 / y_factor;//BRICK_HEIGHT / y_factor;

    BRICK_LAYER_OFFSET = BRICK_LAYER_OFFSET / y_factor;
    BRICK_LAYER_HEIGHT = BRICKS_PER_COLUMN * BRICK_HEIGHT;//BRICK_LAYER_HEIGHT / y_factor;

    y_bits = ceil(log2(GAME_HEIGHT));

    io_printf(IO_BUF, "x:%d, y:%d, bw:%d, bh:%d, blo:%d, blh:%d, xb:%d, bl:%d, u:%d, v:%d, yb:%d\n", x, y, BRICK_WIDTH, BRICK_HEIGHT, BRICK_LAYER_OFFSET, BRICK_LAYER_HEIGHT, x_bat, bat_len, u, v, y_bits);

    // Setup recording
    uint32_t recording_flags = 0;
    if (!recording_initialize(recording_address, &recording_flags))
    {
        rt_error(RTE_SWERR);
        return false;
    }

    io_printf(IO_BUF, "Initialise: completed successfully\n");

    return true;
}

void resume_callback() {
    recording_reset();
}

void timer_callback(uint unused, uint dummy)
{
//    io_printf(IO_BUF, "time = %d", _time);
    use(unused);
    use(dummy);
    // If a fixed number of simulation ticks are specified and these have passed
    //
    //  ticks++;
    //this makes it count twice, WTF!?

    _time++;
    score_change_count++;

    if (!infinite_run && _time >= simulation_ticks)
    {
        io_printf(IO_BUF, "if time = %d\n", _time);
        //spin1_pause();
        recording_finalise();
        // go into pause and resume state to avoid another tick
        simulation_handle_pause_resume(resume_callback);
        //    spin1_callback_off(MC_PACKET_RECEIVED);

        io_printf(IO_BUF, "move count Left %u\n", move_count_l);
        io_printf(IO_BUF, "move count Right %u\n", move_count_r);
        io_printf(IO_BUF, "infinite_run %d; time %d\n",infinite_run, _time);
        io_printf(IO_BUF, "simulation_ticks %d\n",simulation_ticks);
        //    io_printf(IO_BUF, "key count Left %u", left_key_count);
        //    io_printf(IO_BUF, "key count Right %u", right_key_count);

        io_printf(IO_BUF, "Exiting on timer.\n");
        simulation_ready_to_read();

        _time -= 1;
        return;
    }
    // Otherwise
    else
    {
        // Increment ticks in frame counter and if this has reached frame delay
//        io_printf(IO_BUF, "else time = %d\n", _time);
        if(_time % 20 == 0)
        {
            if (!current_number_of_bricks && bricking == 1)
            {
                for (int i =0; i<BRICKS_PER_COLUMN; i++)
                {
                    for (int j=0; j<BRICKS_PER_ROW; j++)
                    {
                        bricks[i][j] = true;
                    }
                }
                current_number_of_bricks = BRICKS_PER_COLUMN * BRICKS_PER_ROW;
                //          print_bricks = true;
                v = -MAX_BALL_SPEED;
                //todo make this random in some respect or not
                y = GAME_HEIGHT - 2;

                if(mars_kiss64_seed(kiss_seed) > 0x7FFFFFFF){
                    //        io_printf(IO_BUF, "MARS 2");
                    u = -u;
                }

                //randomises initial x location
                x = x_bat + (bat_len / 2);
            }

            //       if (print_bricks) {
            //        print_bricks = false;
            for (int i =0; i<BRICKS_PER_COLUMN; i++)
            {
                for (int j=0; j<BRICKS_PER_ROW; j++)
                {
                    if (bricks[i][j])
                    {
//                        io_printf(IO_BUF, "adding brick event at i:%d j:%d\n", i, j);
                        add_event(j * BRICK_WIDTH, (i * BRICK_HEIGHT) + BRICK_LAYER_OFFSET, COLOUR_BRICK_ON, true);
                    }
                }
            }
            // If this is the first update, draw bat as
            // collision detection relies on this
            if(_time == FRAME_DELAY)
            {
//                io_printf(IO_BUF, "sets the bat for the first time bl:%d, xb:%, gh:%d\n", bat_len, x_bat, GAME_HEIGHT);
                // Draw bat
                for (int i = x_bat; i < (x_bat + bat_len); i++)
                {
                    set_pixel_col(i, GAME_HEIGHT-1, COLOUR_BAT, false);
                }
            }
            update_frame(_time);
            // Update recorded score every 1s
            if(score_change_count>=1000){
                recording_record(0, &current_score, 4);
                score_change_count=0;
            }
        }
    }
}

void mc_packet_received_callback(uint key, uint payload)
{
    use(payload);
    // Right
    if(key & KEY_RIGHT){
        right_key_count++;
    }
    // Left
    else {
        left_key_count++;
    }
}
//-------------------------------------------------------------------------------

INT_HANDLER sark_int_han (void);


void rte_handler (uint code)
{
  // Save code

  sark.vcpu->user0 = code;
  sark.vcpu->user1 = (uint) sark.sdram_buf;

  // Copy ITCM to SDRAM

  sark_word_cpy (sark.sdram_buf, (void *) ITCM_BASE, ITCM_SIZE);

  // Copy DTCM to SDRAM

  sark_word_cpy (sark.sdram_buf + ITCM_SIZE, (void *) DTCM_BASE, DTCM_SIZE);

  // Try to re-establish consistent SARK state

  sark_vic_init ();

  sark_vic_set ((vic_slot) sark_vec->sark_slot, CPU_INT, 1, sark_int_han);

  uint *stack = sark_vec->stack_top - sark_vec->svc_stack;

  stack = cpu_init_mode (stack, IMASK_ALL+MODE_IRQ, sark_vec->irq_stack);
  stack = cpu_init_mode (stack, IMASK_ALL+MODE_FIQ, sark_vec->fiq_stack);
  (void)  cpu_init_mode (stack, IMASK_ALL+MODE_SYS, 0);

  cpu_set_cpsr (MODE_SYS);

  // ... and sleep

  while (1)
    cpu_wfi ();
}

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


    // frame buffer: 160 x 128 x 4 bits: [hard/soft, R, G, B]
//    static int frame_buff[GAME_WIDTH / 8][GAME_HEIGHT];

    init_frame();
    keystate = 0; // IDLE
    pkt_count = 0;

    // Set timer tick (in microseconds)
    io_printf(IO_BUF, "setting timer tick callback for %d microseconds\n",
              timer_period);
    spin1_set_timer_tick(timer_period);
    io_printf(IO_BUF, "bricks %x\n", &bricks);

    io_printf(IO_BUF, "simulation_ticks %d\n",simulation_ticks);

    // Register callback
    spin1_callback_on(TIMER_TICK, timer_callback, 2);
    spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);

    _time = UINT32_MAX;

    simulation_run();
}
