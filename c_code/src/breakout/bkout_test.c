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

// Spin 1 API includes
#include <spin1_api.h>

// Common includes
#include <debug.h>

// Front end common includes
#include <data_specification.h>
#include <simulation.h>

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
// **TODO** many of these magic numbers should be passed from Python
// Game dimension constants
#define RES_DIV 4
#define GAME_WIDTH  (160/RES_DIV)
#define GAME_HEIGHT (128/RES_DIV)

#define COLOUR_BITS 1
#define WIDTH_BITS  6
#define HEIGHT_BITS 6
#define HORIZ_SHIFT (HEIGHT_BITS + COLOUR_BITS)
#define VERT_SHIFT  COLOUR_BITS
#define PIXELS_PER_WORD 8 //8*[4-bit colour] = 32-bit


// Ball outof play time (frames)
#define OUT_OF_PLAY 100

// Frame delay (ms)
#define FRAME_DELAY 20

// ball position and velocity scale factor
#define FACT 2

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
typedef enum
{
  REGION_SYSTEM,
  REGION_BREAKOUT,
  REGION_PROVENANCE,
} region_t;

typedef enum
{
  COLOUR_HARD       = 0x8,
  COLOUR_SOFT       = 0x0,

  COLOUR_BACKGROUND = COLOUR_SOFT | 0x1,
  COLOUR_BAT        = COLOUR_HARD | 0x6,
  COLOUR_BALL       = COLOUR_HARD | 0x7,
  COLOUR_SCORE      = COLOUR_SOFT | 0x6,
} colour_t;

typedef enum
{
  KEY_LEFT  = 0x1,
  KEY_RIGHT = 0x2,
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
// initial ball coordinates in fixed-point
static int x = (GAME_WIDTH / 4) * FACT;
static int y = (GAME_HEIGHT / 4) * FACT;

// initial ball velocity in fixed-point
static int u = 1 * FACT;
static int v = -1 * FACT;

// bat LHS x position
static int x_bat   = 40/RES_DIV;

// bat length in pixels
static int bat_len = 16/RES_DIV;

// frame buffer: 160 x 128 x 4 bits: [hard/soft, R, G, B]
static int frame_buff[GAME_WIDTH / 8][GAME_HEIGHT];

// control pause when ball out of play
static int out_of_play = 0;

// state of left/right keys
static int keystate = 0;

//! The upper bits of the key value that model should transmit with
static uint32_t key;

//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

//! the number of timer ticks that this model should run for before exiting.
static uint32_t simulation_ticks = 0;

//! How many ticks until next frame
static uint32_t tick_in_frame = 0;

//----------------------------------------------------------------------------
// Inline functions
//----------------------------------------------------------------------------
static inline void add_score_up_event()
{
  spin1_send_mc_packet(key | SPECIAL_EVENT_SCORE_UP, 0, NO_PAYLOAD);
  log_debug("Score up");
}

static inline void add_score_down_event()
{
  spin1_send_mc_packet(key | SPECIAL_EVENT_SCORE_DOWN, 0, NO_PAYLOAD);
  log_debug("Score down");
}

static inline void add_event(int i, int j, colour_t col)
{
  const uint32_t colour_bit = (col == COLOUR_BACKGROUND) ? 0 : 1;
  const uint32_t spike_key = key | (SPECIAL_EVENT_MAX + (i << HORIZ_SHIFT) \
                                    + (j << VERT_SHIFT) + colour_bit);

  spin1_send_mc_packet(spike_key, 0, NO_PAYLOAD);
  // spin1_delay_us(500);
  // spin1_send_mc_packet(spike_key, 0, NO_PAYLOAD);
  log_debug("%d, %d, %u, %08x", i, j, col, spike_key);
}

// gets pixel colour from within word
static inline colour_t get_pixel_col (int i, int j)
{
  return (colour_t)(frame_buff[i / PIXELS_PER_WORD][j] >> \
                   ((i % PIXELS_PER_WORD)*4) & 0xF);
}

// inserts pixel colour within word
static inline void set_pixel_col (int i, int j, colour_t col)
{
    if (col != get_pixel_col(i, j))
    {
      frame_buff[i / PIXELS_PER_WORD][j] = (frame_buff[i / PIXELS_PER_WORD][j] & \
                                             ~(0xF << ((i % PIXELS_PER_WORD) * 4))) \
                                            | ((int)col << ((i % PIXELS_PER_WORD)*4));
      add_event (i, j, col);
    }
}

//----------------------------------------------------------------------------
// Static functions
//----------------------------------------------------------------------------
// initialise frame buffer to blue
static void init_frame ()
{
  for (int i=0; i<(GAME_WIDTH/PIXELS_PER_WORD); i++)
  {
    for (int j=0; j<GAME_HEIGHT; j++)
    {
      frame_buff[i][j] = 0x11111111 * COLOUR_BACKGROUND;
    }
  }
}

static void update_frame ()
{
// draw bat
  // Cache old bat position
  const int old_xbat = x_bat;

  // Update bat and clamp
  if (keystate & KEY_LEFT && --x_bat < 0)
  {
    x_bat = 0;
  }
  else if (keystate & KEY_RIGHT && ++x_bat > GAME_WIDTH-bat_len)
  {
    x_bat = GAME_WIDTH-bat_len;
  }

  // Clear keystate
  keystate = 0;

  // If bat's moved
  if (old_xbat != x_bat)
  {
    // Draw bat pixels
    for (int i = x_bat; i < (x_bat + bat_len); i++)
    {
      set_pixel_col(i, GAME_HEIGHT-1, COLOUR_BAT);
    }

    // Remove pixels left over from old bat
    if (x_bat > old_xbat)
    {
      set_pixel_col(old_xbat, GAME_HEIGHT-1, COLOUR_BACKGROUND);
    }
    else if (x_bat < old_xbat)
    {
      set_pixel_col(old_xbat + bat_len, GAME_HEIGHT-1, COLOUR_BACKGROUND);
    }
  }

// draw ball
  if (out_of_play == 0)
  {
    // clear pixel to background
    set_pixel_col(x/FACT, y/FACT, COLOUR_BACKGROUND);

    // move ball in x and bounce off sides
    x += u;
    if ((x < -u) || (x >= ((GAME_WIDTH*FACT)-u)))
    {
      u = -u;
    }

    // move ball in y and bounce off top
    y += v;
    if (y < -v)
    {
      v = -v;
    }

//detect collision
    // if we hit something hard!
    if (get_pixel_col(x / FACT, y / FACT) & COLOUR_HARD)
    {
      if (x/FACT < (x_bat+bat_len/4))
      {
        u = -FACT;
      }
      else if (x/FACT < (x_bat+bat_len/2))
      {
        u = -FACT/2;
      }
      else if (x/FACT < (x_bat+3*bat_len/4))
      {
        u = FACT/2;
      }
      else
      {
        u = FACT;
      }

      v = -FACT;
      y -= FACT;

      // Increase score
      add_score_up_event();
    }

// lost ball
    if (y >= (GAME_HEIGHT*FACT-v))
    {
      v = -1 * FACT;
      y = (GAME_HEIGHT / 2)*FACT;
      out_of_play = OUT_OF_PLAY;
      
      // Decrease score
      add_score_down_event();
    }
    // draw ball
    else
    {
      set_pixel_col(x/FACT, y/FACT, COLOUR_BALL);
    }
  }
  else
  {
    --out_of_play;
  }
}

static bool initialize(uint32_t *timer_period)
{
  log_info("Initialise: started");

  // Get the address this core's DTCM data starts at from SRAM
  address_t address = data_specification_get_data_address();

  // Read the header
  if (!data_specification_read_header(address))
  {
      return false;
  }

  // Get the timing details and set up the simulation interface
  if (!simulation_initialise(data_specification_get_region(REGION_SYSTEM, address),
    APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
    &infinite_run, 1, NULL, data_specification_get_region(REGION_PROVENANCE, address)))
  {
      return false;
  }

  // Read breakout region
  address_t breakout_region = data_specification_get_region(REGION_BREAKOUT, address);
  key = breakout_region[0];
  log_info("\tKey=%08x", key);

  log_info("Initialise: completed successfully");

  return true;
}

//----------------------------------------------------------------------------
// Callbacks
//----------------------------------------------------------------------------
// incoming SDP message
/*void process_sdp (uint m, uint port)
*{
    sdp_msg_t *msg = (sdp_msg_t *) m;

    io_printf (IO_BUF, "SDP len %d, port %d - %s\n", msg->length, port, msg->data);
    // Port 1 - key data
    if (port == 1) spin1_memcpy(&keystate, msg->data, 4);
    spin1_msg_free (msg);
    if (port == 7) spin1_exit (0);
}*/

void timer_callback(uint ticks, uint dummy)
{
  // If a fixed number of simulation ticks are specified and these have passed
  // **NOTE** ticks starts at 1!
  if (!infinite_run && (ticks - 1) >= simulation_ticks)
  {
    spin1_pause();
    return;
  }
  // Otherwise
  else
  {
    // Increment ticks in frame counter and if this has reached frame delay
    tick_in_frame++;
    if(tick_in_frame == FRAME_DELAY)
    {
      // If this is the first update, draw bat as
      // collision detection relies on this
      if(ticks == FRAME_DELAY)
      {
        // Draw bat
        for (int i = x_bat; i < (x_bat + bat_len); i++)
        {
          set_pixel_col(i, GAME_HEIGHT-1, COLOUR_BAT);
        }
      }

      // Reset ticks in frame and update frame
      tick_in_frame = 0;
      update_frame();
    }
  }
}

void mc_packet_received_callback(uint key, uint payload)
{
  log_debug("Packet received %08x", key);
  // log_info("Packet received %08x", key);

  // Left
  if(key & 0x1)
  {
    keystate |= KEY_LEFT;
  }
  // Right
  else
  {
    keystate |= KEY_RIGHT;
  }
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
    log_error("Error in initialisation - exiting!");
    rt_error(RTE_SWERR);
    return;
  }

  init_frame();
  keystate = 0;
  tick_in_frame = 0;

  // Set timer tick (in microseconds)
  spin1_set_timer_tick(timer_period);

  // Register callback
  spin1_callback_on(TIMER_TICK, timer_callback, 2);
  spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);

  simulation_run();
}
