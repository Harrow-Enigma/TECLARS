#include <Arduino.h>
#include "TFT_eSPI.h"
#include <Wire.h>

TFT_eSPI tft;

#define BACKCOL TFT_BLACK  // Homescreen background colour
const uint32_t COLORMAP[4] = { TFT_WHITE, TFT_GREEN, TFT_YELLOW, TFT_RED };
bool btn_pressed = false;

// Create project homescreen on Wio
void project(){
  tft.fillScreen(BACKCOL);
  tft.setTextColor(TFT_WHITE);
  tft.setTextSize(3);
  tft.drawString("TECLARS", 35, 30);
  tft.setTextSize(2);
  tft.drawString("by TEAM ENIGMA", 35, 200);
}

void setup() {
  pinMode(WIO_5S_UP, INPUT_PULLUP);
  pinMode(WIO_5S_DOWN, INPUT_PULLUP);
  pinMode(WIO_5S_LEFT, INPUT_PULLUP);
  pinMode(WIO_5S_RIGHT, INPUT_PULLUP);
  pinMode(WIO_5S_PRESS, INPUT_PULLUP);
  pinMode(WIO_KEY_A, INPUT_PULLUP);
  pinMode(WIO_KEY_B, INPUT_PULLUP);
  pinMode(WIO_KEY_C, INPUT_PULLUP);
  
  Serial.begin(115200);

  tft.begin();
  tft.setRotation(3);
  project();
  
  tft.setTextSize(1);
  tft.setTextColor(TFT_CYAN);
  tft.drawString("Initialising... Might take up to 60s", 35, 100);
} 

void loop() {
  if (digitalRead(WIO_5S_PRESS) == LOW) {
    tft.setTextColor(TFT_WHITE);
    tft.setTextSize(2.5);
    tft.drawString("> Picture Captured", 35, 150);
    btn_pressed = true;
  }
  
  if (Serial.available() > 0) {
    // Read string from serial
    String msg = Serial.readStringUntil('\n');
    int colorIdx = Serial.parseInt();
    delay(10);
    Serial.println(btn_pressed);

    // Reset button graphics
    btn_pressed = false;
    
    // Write message received from serial
    tft.fillRect(30, 80, 285, 100, BACKCOL);  // Erase previous outputs
    tft.setTextColor(COLORMAP[colorIdx]);
    tft.setTextSize(2.5);
    tft.drawString(msg, 35, 100);
    
    delay(100);
  
  } else {
    delay(100);
  }
}
