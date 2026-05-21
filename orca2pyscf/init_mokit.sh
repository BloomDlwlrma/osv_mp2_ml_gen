#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/share1/anaconda/3.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/share1/anaconda/3.11/etc/profile.d/conda.sh" ]; then
        . "/share1/anaconda/3.11/etc/profile.d/conda.sh"
    else
        export PATH="/share1/anaconda/3.11/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate /scr/u/u3651388/conda/envs/tdnn

# Load modules required for MOKIT
module load intel/2024.2
module load impi/2022.2

# MOKIT runtime variables (caller may override MOKIT_ROOT)
export MOKIT_ROOT="${MOKIT_ROOT:-$HOME/softwares/mokit}"
export MOKIT_BIN="${MOKIT_BIN:-$MOKIT_ROOT/bin/mkl2fch}"

# Prepend a path segment only once to avoid duplicates when sourced repeatedly.
prepend_once() {
    local segment="$1"
    local current="${2-}"
    if [[ -z "$current" ]]; then
        printf '%s' "$segment"
        return
    fi
    case ":$current:" in
        *":$segment:"*) printf '%s' "$current" ;;
        *) printf '%s' "$segment:$current" ;;
    esac
}

export PATH="$(prepend_once "$MOKIT_ROOT/bin" "${PATH-}")"
export PYTHONPATH="$(prepend_once "$MOKIT_ROOT" "${PYTHONPATH-}")"
