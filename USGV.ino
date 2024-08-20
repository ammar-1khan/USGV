#include "BluetoothSerial.h"
#include <FirebaseESP32.h>
#include <WiFi.h>

#define WIFI_SSID "iPhone"
#define WIFI_PASSWORD "11221122"
#define FIREBASE_HOST "remotecontrol-c54b9-default-rtdb.firebaseio.com"  /// databaser URL
#define FIREBASE_AUTH "8F5hIOvMLRd-8k11cxTY09Lqa0Aqfrgf1RrSv4JzH7Y"   /// API key

//8F5hIOvMLRd-8k11cxTY09Lqa0Aqfrgf1RrSv4JzH7Y

FirebaseData fbdo;
FirebaseData getdata1;

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run make menuconfig to and enable it
#endif

BluetoothSerial SerialBT;

#define LED_GPIO   5
#define PWM1_Ch    0
#define PWM1_Res   8
#define PWM1_Freq  1000

void setup() {
  pinMode(12, OUTPUT);
  pinMode(14, OUTPUT);
  pinMode(27, OUTPUT);
  pinMode(26, OUTPUT);

  digitalWrite(12, 0);
  digitalWrite(14, 0);
  digitalWrite(27, 0);
  digitalWrite(26, 0);

  Serial.begin(115200);
  SerialBT.begin("USGV"); //Bluetooth device name
  Serial.println("The device started, now you can pair it with bluetooth!");

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  Firebase.begin(FIREBASE_HOST, FIREBASE_AUTH);
  Firebase.reconnectWiFi(true);

  ledcAttachPin(LED_GPIO, PWM1_Ch);
  ledcSetup(PWM1_Ch, PWM1_Freq, PWM1_Res);
}
char rx;
bool manualMode = true;
unsigned long lastTimeManual = 0;

void loop() {
  if (SerialBT.available()) {
    rx = SerialBT.read();
    Serial.println(rx);
  }

  if (digitalRead(2) == LOW) {
    manualMode = true;
    lastTimeManual = millis();
  } else {
    manualMode = false;
  }

  if (manualMode) {
    manualControl();
  } else {
    automaticControl();
  }
  delay(20);
}

void manualControl() {
  int s = analogRead(A0);
  if (s > 300) {
    digitalWrite(2, 1);
  } else {
    digitalWrite(2, 0);
  }

  if (rx == 'F') {
    Serial.println("FORWARD");
    ledcWrite(PWM1_Ch, 200);
    digitalWrite(12, 1);
    digitalWrite(14, 0);
    digitalWrite(26, 1);
    digitalWrite(27, 0);
  } else if (rx == 'B') {
    Serial.println("BACKWARD");
    ledcWrite(PWM1_Ch, 200);
    digitalWrite(12, 0);
    digitalWrite(14, 1);
    digitalWrite(26, 0);
    digitalWrite(27, 1);
  } else if (rx == 'R') {
    Serial.println("LEFT");
    ledcWrite(PWM1_Ch, 250);
    digitalWrite(12, 1);
    digitalWrite(14, 0);
    digitalWrite(26, 0);
    digitalWrite(27, 1);
  } else if (rx == 'L') {
    Serial.println("RIGHT");
    ledcWrite(PWM1_Ch, 250);
    digitalWrite(12, 0);
    digitalWrite(14, 1);
    digitalWrite(26, 1);
    digitalWrite(27, 0);
  } else if (rx == 'S') {
    Serial.println("STOP");
    digitalWrite(12, 0);
    digitalWrite(14, 0);
    digitalWrite(26, 0);
    digitalWrite(27, 0);
  }
}

void automaticControl() {
  unsigned long currentTime = millis();
  if ((currentTime - lastTimeManual) % 12000 < 6000) {
    // Forward for 6 seconds
    Serial.println("FORWARD");
    ledcWrite(PWM1_Ch, 200);
    digitalWrite(12, 1);
    digitalWrite(14, 0);
    digitalWrite(26, 1);
    digitalWrite(27, 0);
  } else {
    // Backward for 6 seconds
    Serial.println("BACKWARD");
    ledcWrite(PWM1_Ch, 200);
    digitalWrite(12, 0);
    digitalWrite(14, 1);
    digitalWrite(26, 0);
    digitalWrite(27, 1);
  }
}
