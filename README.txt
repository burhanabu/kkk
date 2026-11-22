ACCENTLIVE PRO - poora setup

1) BANANA (GitHub): zip ki files repo mein upload karein (app.py, core.py, index.html,
   requirements.txt, README.txt) aur .github/workflows/build.yml. Actions -> Run workflow.
   Hara tick aane par Artifacts se "AccentLivePro-windows" download karke unzip karein.
2) VIRTUAL MIC (ek baar): app mein "Install virtual microphone" button dabayein (ya vb-audio.com/Cable),
   Windows prompt allow karein, PC restart.
3) WINDOWS SOUND: Settings -> Sound -> Recording: "CABLE Output" par right-click ->
   "Set as Default Device" aur "Set as Default Communication Device".
   CABLE Output -> Properties -> Listen: "Listen to this device" ka tick HATA dein.
4) APP: Name aur Company likhein (settings khud save hoti hain). "Hear it myself on" = Off.
   Status "Ready. Buttons are prepared." aane par Start call dabayein.
5) USE: button / 1-9 keys = foran awaaz. Text likh kar Enter. "+ Button" se line button ban jati hai.
   Live voice tick karein to apne mic mein bolne par awaaz convert ho kar jati hai (My microphone = apna asli mic).
   Esc = Stop.

Chrome, WhatsApp Web, dialer: jahan mic default hai wahan awaaz khud jati hai; jahan mic chunna padta hai wahan CABLE Output chunein.
Start call par pehle naam, company aur AI voice wali line bolti hai.

NOTE: "models" folder (kokoro.onnx, voices.bin) is for local backup / start.bat use.
GitHub par models upload NAHI karni (badi files hain); build.yml unhe khud download kar leti hai.
