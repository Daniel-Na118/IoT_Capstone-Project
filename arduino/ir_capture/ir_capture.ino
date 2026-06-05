/*
  Usage:
    Serial Monitor at 115200 baud
    Note the "Address" and "Command" values printed.
    Put values into ir_sender.h:
      IR_ON_ADDRESS  / IR_ON_COMMAND
      IR_OFF_ADDRESS / IR_OFF_COMMAND
*/

#include <IRremote.hpp>

#define IR_RECV_PIN 4

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    continue;
  }
  IrReceiver.begin(IR_RECV_PIN, ENABLE_LED_FEEDBACK);

  Serial.print(F("Ready. Listening for IR codes on pin "));
  Serial.println(IR_RECV_PIN);
  Serial.println(F("Press a button on your remote..."));
  Serial.println();
}

void loop() {
  if (!IrReceiver.decode()) return;

  IrReceiver.printIRResultShort(&Serial);

  IrReceiver.printIRSendUsage(&Serial);

  if (IrReceiver.decodedIRData.protocol == UNKNOWN) {
    Serial.println(F("Protocol unknown"));
    IrReceiver.printIRResultRawFormatted(&Serial, true);
  }

  Serial.println();
  IrReceiver.resume();
}
