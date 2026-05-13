#!/bin/bash

# =========================================================
# AV ROBOT CONTROL CENTER v5.4 (Added Object Detection Test)
# =========================================================

# Source the workspace
source ~/slam_ws/devel/setup.bash

# Define Directories
MAP_DIR="/home/rfran/dev/maps"
OSA_DIR="/home/rfran/dev/osa"
TEMP_FILE="/tmp/current_map.txt"

mkdir -p "$MAP_DIR" "$OSA_DIR"

# ANSI Color Codes
RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'
CYAN='\e[36m'
NC='\e[0m'

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

pause() { echo ""; read -p "Press Enter to return to menu..." ; }

kill_ros_nodes() {
    echo -e "${YELLOW}>> Shutting down ROS nodes...${NC}"
    pkill -f roslaunch > /dev/null 2>&1
    pkill -f master_nav.launch > /dev/null 2>&1
    sleep 3 
}

# IMPROVED: This now checks the APP, not just the PHONE
check_camera_link() {
    LAUNCH_FILE="$(rospack find simple_nav)/launch/master_nav.launch"
    if [ ! -f "$LAUNCH_FILE" ]; then return 1; fi
    
    # Extract IP and Port (e.g., 192.168.100.15 and 5000)
    # Added head -n 1 to prevent array issues if multiple matches exist
    IP_ADDR=$(grep 'name="video_stream_provider"' "$LAUNCH_FILE" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
    PORT=$(grep 'name="video_stream_provider"' "$LAUNCH_FILE" | grep -oE ':[0-9]+' | tr -d ':' | head -n 1)
    PORT=${PORT:-5000} # Default to 5000 if not found
    
    echo -e "${YELLOW}>> Checking Video Server at $IP_ADDR:$PORT...${NC}"
    
    # Try to connect to the specific port with a 2-second timeout
    if timeout 2 bash -c "</dev/tcp/$IP_ADDR/$PORT" 2>/dev/null; then
        echo -e "${GREEN}>> Video Server is ACTIVE.${NC}"
        return 0
    else
        echo -e "${RED}>> ERROR: Camera App Not Responding!${NC}"
        echo -e "${RED}>> Make sure the IP Camera app is running and 'Start Server' is clicked.${NC}"
        return 1
    fi
}

# =========================================================
# SETTINGS SUB-MENU
# =========================================================

robot_settings() {
    while true; do
        clear
        echo -e "${CYAN}=========== SYSTEM SETTINGS ===========${NC}"
        echo "[1] Update IP Camera URL"
        echo "[2] Test Camera Connection (Deep Check)"
        echo "[0] Back to Main Menu"
        echo -e "${CYAN}=======================================${NC}"
        read -p "Choose an option: " s_opt
        
        LAUNCH_FILE="$(rospack find simple_nav)/launch/master_nav.launch"
        case $s_opt in
            1)
                clear
                CURRENT_URL=$(grep 'name="video_stream_provider"' "$LAUNCH_FILE" | sed -E 's/.*value="(.*)".*/\1/')
                echo -e "Current URL: ${GREEN}$CURRENT_URL${NC}"
                read -p "Enter new URL (or 'c' to cancel): " n_url
                if [[ "$n_url" != "c" && -n "$n_url" ]]; then
                    # FIXED: Using capture groups \1 and \2 to preserve the exact XML structure including the /> closing tag
                    sed -i -E 's|(name="video_stream_provider"[ \t]*value=")[^"]*(")|\1'"$n_url"'\2|' "$LAUNCH_FILE"
                    echo -e "${GREEN}>> URL updated successfully.${NC}"
                fi
                pause ;;
            2) check_camera_link; pause ;;
            0) return ;;
        esac
    done
}

# =========================================================
# MAIN LOOP 
# =========================================================

while true; do
    clear
    echo -e "${CYAN}=========================================${NC}"
    echo -e "${CYAN}         AV ROBOT CONTROL CENTER         ${NC}"
    echo -e "${CYAN}=========================================${NC}"
    echo "[1] Start Mapping & Integrated Save"
    echo "[2] Load Map & Navigate with Object Detection (Auto-Localize)"
    # echo "[3] Test Object Detection"
    echo "[3] List & Preview Maps"
    echo "[4] Rename / Copy Map"
    echo "[5] Delete Map"
    echo "[6] Clear ROS Logs"
    echo "[7] Settings"
    echo "[0] Exit"
    echo -e "${CYAN}=========================================${NC}"
    read -p "Choose an option (0-8): " option

    case $option in
        1)
            clear
            if check_camera_link; then
                DEFAULT_MAP="MAP-$(date +%b-%d-%H%M%S | tr '[:lower:]' '[:upper:]')"
                read -p "Enter map name (Enter for '${DEFAULT_MAP}', or 'c' to cancel): " map_name
                if [[ "$map_name" == "c" || "$map_name" == "C" ]]; then continue; fi
                map_name=${map_name:-$DEFAULT_MAP}

                gnome-terminal -- bash -c "source ~/slam_ws/devel/setup.bash && roslaunch simple_nav master_nav.launch enable_mapping:=true enable_nav:=false"
                echo ""
                echo -e "${CYAN}====================================================${NC}"
                echo -e " MAPPING IS ACTIVE"
                echo -e "${CYAN}====================================================${NC}"
                read -p "Press [Enter] to SAVE the map, or 'c' to cancel: " choice

                if [[ "$choice" == "c" || "$choice" == "C" ]]; then
                    kill_ros_nodes; pause; continue
                fi

                rosrun map_server map_saver -f "$MAP_DIR/$map_name"

                # Send SIGINT to roslaunch to simulate Ctrl+C → triggers .osa save
                echo -e "${YELLOW}>> Sending shutdown signal to trigger Atlas save...${NC}"
                pkill -SIGINT -f roslaunch > /dev/null 2>&1
                pkill -SIGINT -f master_nav.launch > /dev/null 2>&1

                # Wait for .osa to appear
                echo -e "${YELLOW}>> Waiting for 3D Atlas (.osa)...${NC}"
                OSA_FILE="$OSA_DIR/final_map.osa"
                TIMEOUT=180
                elapsed=0
                while [ ! -f "$OSA_FILE" ] && [ $elapsed -lt $TIMEOUT ]; do
                    echo -ne "  Checking... $elapsed/$TIMEOUT seconds\r"
                    sleep 1
                    elapsed=$((elapsed+1))
                done
                echo ""

                # Hard kill anything still running
                kill_ros_nodes

                if [ -f "$OSA_FILE" ]; then
                    [ -f "$OSA_DIR/$map_name.osa" ] && rm -f "$OSA_DIR/$map_name.osa"
                    mv "$OSA_FILE" "$OSA_DIR/$map_name.osa"
                    echo -e "${GREEN}>> 2D and 3D Atlas saved as: $map_name${NC}"
                else
                    echo -e "${YELLOW}>> 2D map saved successfully.${NC}"
                    echo -e "${YELLOW}>> No 3D atlas detected within timeout.${NC}"
                fi

                pause
            else
                pause
            fi ;;

        2)
            clear
            if check_camera_link; then

                shopt -s nullglob
                maps=("$MAP_DIR"/*.yaml)

                if [ ${#maps[@]} -eq 0 ]; then
                    echo -e "${RED}No maps available.${NC}"
                    pause
                    continue
                fi

                PAGE_SIZE=9
                page=0
                total=${#maps[@]}

                while true; do
                    clear
                    echo -e "${CYAN}=========== SELECT MAP (Page $((page+1))) ===========${NC}"

                    start=$((page * PAGE_SIZE))
                    end=$((start + PAGE_SIZE))

                    if [ $end -gt $total ]; then
                        end=$total
                    fi

                    index=1
                    for ((i=start; i<end; i++)); do
                        base=$(basename "${maps[$i]}" .yaml)

                        PGM_S=$([ -f "$MAP_DIR/$base.pgm" ] && echo -e "${GREEN}[PGM]${NC}" || echo -e "${RED}[NO PGM]${NC}")
                        OSA_S=$([ -f "$OSA_DIR/$base.osa" ] && echo -e "${GREEN}[3D]${NC}" || echo -e "${YELLOW}[2D]${NC}")

                        echo -e "[$index] $base  $PGM_S $OSA_S"

                        index=$((index+1))
                    done

                    echo ""
                    echo "[n] Next page"
                    echo "[b] Previous page"
                    echo "[c] Cancel"
                    echo ""

                    read -p "Select map number: " choice

                    case $choice in

                        n|N)
                            if (( end < total )); then
                                page=$((page+1))
                            fi
                            ;;

                        b|B)
                            if (( page > 0 )); then
                                page=$((page-1))
                            fi
                            ;;

                        c|C)
                            break
                            ;;

                        *)
                            if [[ "$choice" =~ ^[0-9]+$ ]]; then

                                selected=$((start + choice - 1))

                                if (( selected >= start && selected < end )); then

                                    load_file="${maps[$selected]}"
                                    load_name=$(basename "$load_file" .yaml)

                                    echo ""
                                    echo -e "${GREEN}Loading map: $load_name${NC}"

                                    if [ -f "$OSA_DIR/$load_name.osa" ]; then
                                        cp "$OSA_DIR/$load_name.osa" "$OSA_DIR/final_map.osa"
                                    fi

                                    gnome-terminal -- bash -c \
                                    "source ~/slam_ws/devel/setup.bash && \
                                    roslaunch simple_nav master_nav.launch \
                                    enable_mapping:=false \
                                    enable_nav:=true \
                                    enable_perception:=true \
                                    publish_debug_img:=true \
                                    map_yaml:=\"$MAP_DIR/$load_name.yaml\" \
                                    map_pgm:=\"$MAP_DIR/$load_name.pgm\""

                                    echo -e "${YELLOW}>> Waiting for nodes to initialize...${NC}"
                                    sleep 8

                                    echo -e "${YELLOW}>> Return here and press Enter to STOP.${NC}"
                                    read -p ""

                                    kill_ros_nodes

                                    break
                                else
                                    echo -e "${RED}Invalid selection.${NC}"
                                    sleep 1
                                fi
                            fi
                            ;;
                    esac
                done

            else
                pause
            fi ;;
        # 3)
        #     clear
        #     if check_camera_link; then
        #         echo -e "${CYAN}====================================================${NC}"
        #         echo -e " STARTING OBJECT DETECTION TEST"
        #         echo -e "${CYAN}====================================================${NC}"
        #         echo -e "${YELLOW}>> Launching Camera and Perception Node...${NC}"
                
        #         gnome-terminal -- bash -c \
        #         "source ~/slam_ws/devel/setup.bash && \
        #         roslaunch simple_nav master_nav.launch \
        #         enable_mapping:=false \
        #         enable_nav:=false \
        #         enable_perception:=true \
        #         publish_debug_img:=true"
                
        #         echo -e "${YELLOW}>> Waiting for nodes to initialize...${NC}"
        #         sleep 5
                
        #         echo -e "${GREEN}>> Object Detection is running.${NC}"
        #         echo -e "${YELLOW}>> You can view the bounding boxes in RViz via the Image display,${NC}"
        #         echo -e "${YELLOW}>> or by opening a new terminal and running: rqt_image_view${NC}"
        #         echo ""
        #         echo -e "${YELLOW}>> Return here and press Enter to STOP.${NC}"
        #         read -p ""
                
        #         kill_ros_nodes
        #     else
        #         pause
        #     fi ;;
        3)
            clear

            shopt -s nullglob
            maps=("$MAP_DIR"/*.yaml)

            if [ ${#maps[@]} -eq 0 ]; then
                echo -e "${RED}No maps available.${NC}"
                pause
                continue
            fi

            PAGE_SIZE=9
            page=0
            total=${#maps[@]}

            while true; do
                clear
                echo -e "${CYAN}=========== MAP GALLERY & PREVIEW (Page $((page+1))) ===========${NC}"

                start=$((page * PAGE_SIZE))
                end=$((start + PAGE_SIZE))

                if [ $end -gt $total ]; then
                    end=$total
                fi

                index=1

                for ((i=start; i<end; i++)); do
                    base=$(basename "${maps[$i]}" .yaml)

                    PGM_S=$([ -f "$MAP_DIR/$base.pgm" ] && echo -e "${GREEN}[PGM OK]${NC}" || echo -e "${RED}[NO PGM]${NC}")
                    YAML_S=$([ -f "$MAP_DIR/$base.yaml" ] && echo -e "${GREEN}[YAML OK]${NC}" || echo -e "${RED}[NO YAML]${NC}")
                    OSA_S=$([ -f "$OSA_DIR/$base.osa" ] && echo -e "${GREEN}[OSA OK]${NC}" || echo -e "${YELLOW}[NO OSA]${NC}")

                    echo -e "[$index] $base | $YAML_S $PGM_S $OSA_S"

                    index=$((index+1))
                done

                echo ""
                echo "[n] Next page"
                echo "[b] Previous page"
                echo "[c] Cancel"
                echo ""

                read -p "Preview map number: " choice

                case $choice in

                    n|N)
                        if (( end < total )); then
                            page=$((page+1))
                        fi
                        ;;

                    b|B)
                        if (( page > 0 )); then
                            page=$((page-1))
                        fi
                        ;;

                    c|C)
                        break
                        ;;

                    *)
                        if [[ "$choice" =~ ^[0-9]+$ ]]; then

                            selected=$((start + choice - 1))

                            if (( selected >= start && selected < end )); then

                                preview_file="${maps[$selected]}"
                                preview_name=$(basename "$preview_file" .yaml)

                                if [ -f "$MAP_DIR/$preview_name.pgm" ]; then

                                    eog "$MAP_DIR/$preview_name.pgm" & > /dev/null 2>&1

                                    echo ""
                                    echo -e "${YELLOW}>> Press Enter to CLOSE the map preview...${NC}"

                                    read -p ""

                                    pkill -f eog

                                else
                                    echo -e "${RED}PGM file not found.${NC}"
                                    sleep 1
                                fi

                            else
                                echo -e "${RED}Invalid selection.${NC}"
                                sleep 1
                            fi
                        fi
                        ;;
                esac
            done ;;

        4)
            clear

            shopt -s nullglob
            maps=("$MAP_DIR"/*.yaml)

            if [ ${#maps[@]} -eq 0 ]; then
                echo -e "${RED}No maps available.${NC}"
                pause
                continue
            fi

            PAGE_SIZE=9
            page=0
            total=${#maps[@]}

            while true; do
                clear
                echo -e "${CYAN}=========== RENAME / COPY MAP (Page $((page+1))) ===========${NC}"

                start=$((page * PAGE_SIZE))
                end=$((start + PAGE_SIZE))

                if [ $end -gt $total ]; then
                    end=$total
                fi

                index=1

                for ((i=start; i<end; i++)); do

                    base=$(basename "${maps[$i]}" .yaml)

                    PGM_S=$([ -f "$MAP_DIR/$base.pgm" ] && echo -e "${GREEN}[PGM]${NC}" || echo -e "${RED}[NO PGM]${NC}")
                    OSA_S=$([ -f "$OSA_DIR/$base.osa" ] && echo -e "${GREEN}[OK]${NC}" || echo -e "${YELLOW}[2D]${NC}")

                    echo -e "[$index] $base  $PGM_S $OSA_S"

                    index=$((index+1))

                done

                echo ""
                echo "[n] Next page"
                echo "[b] Previous page"
                echo "[c] Cancel"
                echo ""

                read -p "Select map number to rename/copy: " choice

                case $choice in

                    n|N)
                        if (( end < total )); then
                            page=$((page+1))
                        fi
                        ;;

                    b|B)
                        if (( page > 0 )); then
                            page=$((page-1))
                        fi
                        ;;

                    c|C)
                        break
                        ;;

                    *)
                        if [[ "$choice" =~ ^[0-9]+$ ]]; then

                            selected=$((start + choice - 1))

                            if (( selected >= start && selected < end )); then

                                old_file="${maps[$selected]}"
                                old=$(basename "$old_file" .yaml)

                                read -p "New name: " new

                                if [ -z "$new" ]; then
                                    echo -e "${RED}Invalid name.${NC}"
                                    sleep 1
                                    continue
                                fi

                                cp "$MAP_DIR/$old.yaml" "$MAP_DIR/$new.yaml"
                                cp "$MAP_DIR/$old.pgm" "$MAP_DIR/$new.pgm"

                                if [ -f "$OSA_DIR/$old.osa" ]; then
                                    cp "$OSA_DIR/$old.osa" "$OSA_DIR/$new.osa"
                                fi

                                sed -i "s/$old.pgm/$new.pgm/g" \
                                "$MAP_DIR/$new.yaml"

                                echo -e "${GREEN}>> Map copied successfully.${NC}"

                                pause
                                break

                            else
                                echo -e "${RED}Invalid selection.${NC}"
                                sleep 1
                            fi

                        fi
                        ;;
                esac

            done
        ;;

        5)
            clear

            shopt -s nullglob
            maps=("$MAP_DIR"/*.yaml)

            if [ ${#maps[@]} -eq 0 ]; then
                echo -e "${RED}No maps available.${NC}"
                pause
                continue
            fi

            PAGE_SIZE=9
            page=0
            total=${#maps[@]}

            while true; do
                clear
                echo -e "${CYAN}=========== DELETE MAP (Page $((page+1))) ===========${NC}"

                start=$((page * PAGE_SIZE))
                end=$((start + PAGE_SIZE))

                if [ $end -gt $total ]; then
                    end=$total
                fi

                index=1

                for ((i=start; i<end; i++)); do

                    base=$(basename "${maps[$i]}" .yaml)

                    PGM_S=$([ -f "$MAP_DIR/$base.pgm" ] && echo -e "${GREEN}[PGM]${NC}" || echo -e "${RED}[NO PGM]${NC}")
                    OSA_S=$([ -f "$OSA_DIR/$base.osa" ] && echo -e "${GREEN}[3D]${NC}" || echo -e "${YELLOW}[2D]${NC}")

                    echo -e "[$index] $base  $PGM_S $OSA_S"

                    index=$((index+1))

                done

                echo ""
                echo "[n] Next page"
                echo "[b] Previous page"
                echo "[c] Cancel"
                echo ""

                read -p "Select map number to delete: " choice

                case $choice in

                    n|N)
                        if (( end < total )); then
                            page=$((page+1))
                        fi
                        ;;

                    b|B)
                        if (( page > 0 )); then
                            page=$((page-1))
                        fi
                        ;;

                    c|C)
                        break
                        ;;

                    *)
                        if [[ "$choice" =~ ^[0-9]+$ ]]; then

                            selected=$((start + choice - 1))

                            if (( selected >= start && selected < end )); then

                                d_file="${maps[$selected]}"
                                d_name=$(basename "$d_file" .yaml)

                                echo ""
                                read -p "Delete '$d_name'? (yes/no): " confirm

                                if [[ "$confirm" == "yes" ]]; then

                                    rm -f \
                                    "$MAP_DIR/$d_name.yaml" \
                                    "$MAP_DIR/$d_name.pgm" \
                                    "$OSA_DIR/$d_name.osa"

                                    echo -e "${RED}>> Map deleted.${NC}"

                                fi

                                pause
                                break

                            else
                                echo -e "${RED}Invalid selection.${NC}"
                                sleep 1
                            fi

                        fi
                        ;;
                esac

            done
        ;;

        6) rosclean purge -y && sleep 1 ;;
        7) robot_settings ;;
        0) break ;;
    esac
done
