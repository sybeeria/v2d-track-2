#!/usr/bin/env bash
# Usage: scripts/vast/install_sentinel.sh [--remove]
# Runs scripts/vast/sentinel.py every 5 min as a launchd agent; alerts go to the Telegram bot in
# ~/.config/v2d/telegram.env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID). Log: ~/.config/v2d/sentinel/sentinel.log.
set -euo pipefail
label=com.v2d.vast-sentinel
plist=$HOME/Library/LaunchAgents/$label.plist
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
if [ "${1:-}" = --remove ]; then rm -f "$plist"; exit 0; fi
test -s "$HOME/.config/v2d/telegram.env" || { echo "missing $HOME/.config/v2d/telegram.env" >&2; exit 1; }
script=$(cd "$(dirname "$0")" && pwd)/sentinel.py
python=$(command -v python3)
mkdir -p "$HOME/.config/v2d/sentinel" "$(dirname "$plist")"
cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>$python</string><string>-I</string><string>$script</string></array>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>$(dirname "$(command -v vastai)"):/usr/bin:/bin:/usr/sbin</string></dict>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$HOME/.config/v2d/sentinel/sentinel.log</string>
  <key>StandardErrorPath</key><string>$HOME/.config/v2d/sentinel/sentinel.log</string>
</dict></plist>
EOF
plutil -lint "$plist" > /dev/null
launchctl bootstrap "gui/$(id -u)" "$plist"
echo "installed $label: $python -I $script every 300 s"
