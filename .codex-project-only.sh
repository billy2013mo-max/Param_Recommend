#!/bin/sh
set -eu

allowed_root='/wanqing-develop/luowenjing/Param_Recommend'
project_codex_home='/root/.codex-project-homes/Param_Recommend'
default_codex_home='/root/.codex'
actual_codex='/root/.codex/packages/standalone/current/bin/codex'

effective_dir=$(pwd -P 2>/dev/null || true)
requested_dir=''
expect_requested_dir=false

# -C/--cd 改变 Codex 的工作目录，因此账号选择也跟随它的目标目录。
for argument in "$@"; do
    if [ "$expect_requested_dir" = true ]; then
        requested_dir=$argument
        expect_requested_dir=false
        continue
    fi

    case "$argument" in
        --)
            break
            ;;
        -C|--cd)
            expect_requested_dir=true
            ;;
        --cd=*)
            requested_dir=${argument#--cd=}
            ;;
        -C?*)
            requested_dir=${argument#-C}
            ;;
    esac
done

if [ "$expect_requested_dir" = true ]; then
    effective_dir=''
elif [ -n "$requested_dir" ]; then
    case "$requested_dir" in
        /*) target_dir=$requested_dir ;;
        *) target_dir=${effective_dir:+$effective_dir/}$requested_dir ;;
    esac

    if [ -d "$target_dir" ]; then
        effective_dir=$(realpath -e -- "$target_dir" 2>/dev/null || true)
    else
        effective_dir=''
    fi
fi

case "$effective_dir" in
    "$allowed_root"|"$allowed_root"/*)
        CODEX_HOME=$project_codex_home
        ;;
    *)
        CODEX_HOME=$default_codex_home
        ;;
esac

export CODEX_HOME
exec "$actual_codex" -c 'cli_auth_credentials_store="file"' "$@"
