#!/bin/tcsh

# Read scan start/end times from file and compute adjusted times
set time_file = "scan_start_end_times.txt"

if (! -f $time_file) then
    echo "ERROR: $time_file not found. Exiting."
    exit 1
endif

# Extract start and end timestamps from the file
set raw_start = `grep '^Start:' $time_file | awk '{print $2 " " $3}'`
set raw_end = `grep '^End:' $time_file | awk '{print $2 " " $3}'`

echo $raw_start;
echo $raw_end;

# Subtract 1 minute from start time and truncate seconds
set start_epoch = `date -d "$raw_start" +%s`
@ start_epoch_adj = "$start_epoch" - 60
set start_time = `date -d @$start_epoch_adj "+%m/%d/%Y %H:%M:00"`

# Add 2 minutes to end time and truncate seconds
set end_epoch = `date -d "$raw_end" +%s`
@ end_epoch_adj = "$end_epoch" + 120
set end_time = `date -d @$end_epoch_adj "+%m/%d/%Y %H:%M:00"`

echo $start_time;
echo $end_time;

set interval_seconds = 10  # Interval in seconds for the looped exports

# Source login setup
source ~/.login

# Convert start/end to Unix timestamps
set start_ts = `date -d "$start_time" +%s`
set end_ts = `date -d "$end_time" +%s`

# Export data over full interval
exportLoggerData -loggers "RHIC/Instrumentation/BPM/PHENIX/bip8_bx_mon.logreq,RHIC/Instrumentation/BPM/PHENIX/yip8_bx_mon.logreq" \
-start "$start_time" -stop "$end_time" > bpms.dat

exportLoggerData -loggers "RHIC/Instrumentation/BPM/PHENIX/bip8_bx_mon.logreq,RHIC/Instrumentation/BPM/PHENIX/yip8_bx_mon.logreq" \
-expressions "BH8CrossingAngle=-(cell(b7bx_h)-cell(b8bx_h))/16250 YH8CrossingAngle=-(cell(y7bx_h)-cell(y8bx_h))/16250 GH8CrossingAngle=BH8CrossingAngle-YH8CrossingAngle" \
-start "$start_time" -stop "$end_time" > crossing_angle.dat

exportLoggerData -loggers RHIC/Instrumentation/WCM/wcmBunchLengthMan/circulatingData.logreq \
-arrayform full -cells "yel.WCM.bunchLength:circulatingBunchWidthM[.],blu.WCM.bunchLength:circulatingBunchWidthM[.]" \
-start "$start_time" -stop "$end_time" > bunch_widths.dat

exportLoggerData -loggers RHIC/Instrumentation/WCM/wcmBunchLengthMan/circulatingData.logreq \
-cells "yel.WCM.bunchLength:avgCirculatingBunchWidthM,blu.WCM.bunchLength:avgCirculatingBunchWidthM" \
-start "$start_time" -stop "$end_time" > avg_widths.dat

exportLoggerData -loggers "RHIC/Instrumentation/WCM/Blue/WCMdata.logreq,RHIC/Instrumentation/BCMorDCCT/Blue/allReadings.logreq" \
-cells "blu_TotalIons,beamIons" -start "$start_time" -stop "$end_time" > blue_ions.dat

exportLoggerData -loggers "RHIC/Instrumentation/WCM/Yellow/WCMdata.logreq,RHIC/Instrumentation/BCMorDCCT/Yellow/allReadings.logreq" \
-cells "yel_TotalIons,beamIons" -start "$start_time" -stop "$end_time" > yellow_ions.dat

exportLoggerData -loggers "RHIC/Instrumentation/IPM/Emittance.logreq" \
-start "$start_time" -stop "$end_time" > emittance.dat


# Loop for per-interval 1-second data grabs
@ current_ts = $start_ts

while ( $current_ts < $end_ts )
    @ next_ts = $current_ts + 1

    set formatted_start = `date -d "@$current_ts" "+%m/%d/%Y %H:%M:%S"`
    set formatted_stop = `date -d "@$next_ts" "+%m/%d/%Y %H:%M:%S"`
    set timestamp_suffix = `date -d "@$current_ts" "+%y_%H_%M_%S"`

    echo "Exporting profile data for $formatted_start..."

    exportLoggerData -loggers RHIC/Instrumentation/WCM/Blue/ProfileData.logreq \
    -arrayform full -cells "bo2-wcm3:profileDataM:value[.]" \
    -start "$formatted_start" -stop "$formatted_stop" > "blue_profile_${timestamp_suffix}.dat"

    exportLoggerData -loggers RHIC/Instrumentation/WCM/Yellow/ProfileData.logreq \
    -arrayform full -cells "yi2-wcm3:profileDataM[.]" \
    -start "$formatted_start" -stop "$formatted_stop" > "yellow_profile_${timestamp_suffix}.dat"

    @ current_ts += ( $interval_seconds )
end

mkdir -p profiles
mv *_profile_*.dat profiles/

