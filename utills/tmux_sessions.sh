#!/bin/bash

# ヘッダーを出力
printf "%-20s %-20s %-20s\n" "Session Name" "Last Attached Time" "Creation Time"

# セッションリストを取得して整形
tmux list-sessions -F "#{session_name}|#{session_last_attached: %Y-%m-%d %H:%M:%S}|#{session_created: %Y-%m-%d %H:%M:%S}" | while IFS="|" read -r session_name last_attached_time creation_time; do
  # 空のフィールドにデフォルト値を設定
  last_attached_time=${last_attached_time:-"Never Attached"}
  creation_time=${creation_time:-"Unknown"}
  printf "%-20s %-20s %-20s\n" "$session_name" "$last_attached_time" "$creation_time"
done
