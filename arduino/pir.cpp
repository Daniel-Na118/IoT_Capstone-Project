#include "pir.h"
#include "Arduino.h"

static int g_pir_pin = -1;

void PIR_init(int pin) {
  g_pir_pin = pin;
  pinMode(pin, INPUT);
}

bool PIR_motionDetected() {
  if (g_pir_pin < 0) return false;
  return digitalRead(g_pir_pin) == HIGH;
}
