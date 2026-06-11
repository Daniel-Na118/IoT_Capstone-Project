#include "ir_sender.h"
#include "Arduino.h"
#include <IRremote.hpp>

void IR_init() {
  IrSender.begin(IR_SEND_PIN);
}

void IR_sendOn() {
  IrSender.sendNEC(IR_ON_ADDRESS, IR_ON_COMMAND, 0);
}

void IR_sendOff() {
  IrSender.sendNEC(IR_OFF_ADDRESS, IR_OFF_COMMAND, 0);
}
