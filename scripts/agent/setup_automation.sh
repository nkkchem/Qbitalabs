#!/bin/bash
# QBitaLabs Automation Setup
# Configures cron/launchd for automated nightly agent runs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "============================================"
echo "QBitaLabs Automation Setup"
echo "============================================"
echo ""

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "Detected: macOS (using launchd)"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "Detected: Linux (using cron)"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Function to setup launchd on macOS
setup_launchd() {
    PLIST_NAME="com.qbitalabs.dailyagent"
    PLIST_PATH="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"

    echo "Creating launchd plist..."

    cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$PLIST_NAME</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PROJECT_ROOT/scripts/agent/daily_agent.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$PROJECT_ROOT</string>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>0</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>$PROJECT_ROOT/mvp/logs/launchd_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>$PROJECT_ROOT/mvp/logs/launchd_stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
EOF

    echo "Loading launchd job..."
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    launchctl load "$PLIST_PATH"

    echo ""
    echo "✓ Launchd job installed!"
    echo "  Plist: $PLIST_PATH"
    echo "  Schedule: Daily at midnight"
    echo ""
    echo "Commands:"
    echo "  Start now:  launchctl start $PLIST_NAME"
    echo "  Stop:       launchctl stop $PLIST_NAME"
    echo "  Unload:     launchctl unload $PLIST_PATH"
    echo "  Status:     launchctl list | grep qbitalabs"
}

# Function to setup cron on Linux
setup_cron() {
    CRON_CMD="0 0 * * * cd $PROJECT_ROOT && $PROJECT_ROOT/scripts/agent/daily_agent.sh >> $PROJECT_ROOT/mvp/logs/cron.log 2>&1"

    echo "Adding cron job..."

    # Check if job already exists
    if crontab -l 2>/dev/null | grep -q "qbitalabs"; then
        echo "Removing existing QBitaLabs cron job..."
        crontab -l | grep -v "qbitalabs" | crontab -
    fi

    # Add new job
    (crontab -l 2>/dev/null; echo "# QBitaLabs Daily Agent"; echo "$CRON_CMD") | crontab -

    echo ""
    echo "✓ Cron job installed!"
    echo "  Schedule: Daily at midnight (0 0 * * *)"
    echo ""
    echo "Commands:"
    echo "  View jobs:   crontab -l"
    echo "  Edit jobs:   crontab -e"
    echo "  Remove:      crontab -l | grep -v qbitalabs | crontab -"
}

# Make scripts executable
echo "Making scripts executable..."
chmod +x "$PROJECT_ROOT/scripts/agent/daily_agent.sh"
chmod +x "$PROJECT_ROOT/scripts/train/train_mvp.sh"
chmod +x "$PROJECT_ROOT/scripts/data/download_all.sh"
chmod +x "$PROJECT_ROOT/scripts/demo/investor_demo.sh"
chmod +x "$PROJECT_ROOT/scripts/setup_mvp.sh"
echo "✓ Scripts are executable"
echo ""

# Setup based on OS
if [[ "$OS" == "macos" ]]; then
    read -p "Install launchd job for nightly runs? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_launchd
    else
        echo "Skipped launchd installation"
        echo ""
        echo "To run manually:"
        echo "  $PROJECT_ROOT/scripts/agent/daily_agent.sh"
    fi
elif [[ "$OS" == "linux" ]]; then
    read -p "Install cron job for nightly runs? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_cron
    else
        echo "Skipped cron installation"
        echo ""
        echo "To run manually:"
        echo "  $PROJECT_ROOT/scripts/agent/daily_agent.sh"
    fi
fi

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
