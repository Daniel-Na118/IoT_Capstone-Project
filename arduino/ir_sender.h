#ifndef IR_SENDER_H_
#define IR_SENDER_H_

#define IR_SEND_PIN 3

// NEC protocol codes for light
#define IR_ON_ADDRESS  0xF7  // TODO: replace on-address
#define IR_ON_COMMAND  0xE8  // TODO: replace on-command
#define IR_OFF_ADDRESS 0xF7  // TODO: replace off-address
#define IR_OFF_COMMAND 0xC0  // TODO: replace off-command

void IR_init();
void IR_sendOn();
void IR_sendOff();

#endif  // IR_SENDER_H_
