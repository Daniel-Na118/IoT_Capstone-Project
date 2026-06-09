#include <IRremote.hpp>
#include "USB.h"
#include "USBHIDKeyboard.h"

#define IR_RECV_PIN    15

#define IR_ON_ADDRESS  0x55
#define IR_ON_COMMAND  0x2D
#define IR_OFF_ADDRESS 0x55
#define IR_OFF_COMMAND 0xA5

#define UNLOCK_PASSWORD "placeholder"

#define POST_UNLOCK_DELAY_MS 1200
#define COOLDOWN_MS 3000

USBHIDKeyboard Keyboard;

static unsigned long g_last_action_ms = 0;

static void press_combo(uint8_t mod, uint8_t key) {
  Keyboard.press(mod);
  Keyboard.press(key);
  delay(80);
  Keyboard.releaseAll();
}

void setup() {
  USB.begin();
  Keyboard.begin();
  IrReceiver.begin(IR_RECV_PIN, DISABLE_LED_FEEDBACK);
}

void loop() {
  if (!IrReceiver.decode()) return;

  uint16_t addr = IrReceiver.decodedIRData.address;
  uint8_t  cmd  = IrReceiver.decodedIRData.command;
  unsigned long now = millis();

  if (now - g_last_action_ms >= COOLDOWN_MS) {
    if (addr == IR_ON_ADDRESS && cmd == IR_ON_COMMAND) {
      g_last_action_ms = now;

      delay(150);
      Keyboard.print(UNLOCK_PASSWORD);
      Keyboard.press(KEY_RETURN);
      delay(80);
      Keyboard.releaseAll();

      delay(POST_UNLOCK_DELAY_MS);
      press_combo(KEY_LEFT_GUI, 't');

    } else if (addr == IR_OFF_ADDRESS && cmd == IR_OFF_COMMAND) {
      g_last_action_ms = now;

      press_combo(KEY_LEFT_GUI, 'l');
    }
  }

  IrReceiver.resume();
}
