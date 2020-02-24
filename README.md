# Tapiocas
Tap I/O Calibrated Automaton System  
ie. python adb helper to automate tasks

## FAQ
### useful terminal commands:
connect to phone via usb:  
`adb start-server`

list connected devices  
`adb devices`

listen to events  
`adb shell -- getevent -lt;`

Then press the phone's screen and get data like:  
[ 1059536.835861] /dev/input/event2: EV_KEY       BTN_TOUCH            DOWN  
[ 1059536.835861] /dev/input/event2: EV_ABS       ABS_MT_TRACKING_ID   0000eb79  
[ 1059536.835861] /dev/input/event2: EV_ABS       ABS_MT_POSITION_X    00000213  
[ 1059536.835861] /dev/input/event2: EV_ABS       ABS_MT_POSITION_Y    00000420  
=> "/dev/input/event2" will be needed to send events

kill the server before using this module. May not be able to connect through python otherwise  
`adb kill-server`


### KitKat+ devices require authentication, see this link in case it's not automatic with adb_shell library
https://stackoverflow.com/questions/33005354/trouble-installing-m2crypto-with-pip-on-os-x-macos

### connection over wifi
https://futurestud.io/tutorials/how-to-debug-your-android-app-over-wifi-without-root
