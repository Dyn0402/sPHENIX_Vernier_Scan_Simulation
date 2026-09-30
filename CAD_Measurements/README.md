# CAD Measurements

`get_cad_measurements.tcsh` pulls beam data (crossing angle, BPMs, bunch widths, intensities, emittance, longitudinal profiles) from the RHIC CAD logger database for a given time window, typically a Vernier scan.

## Requirement: run on a machine with CAD database access

The script calls `exportLoggerData`, the CAD logger command-line tool, and does `source ~/.login` to set it up. **It only works on a machine that has the CAD logger tools and can reach the CAD database**, i.e. an sPHENIX/RCF (BNL) machine, not a laptop or an offsite machine. Check with:

```bash
which exportLoggerData
```

If that isn't found after logging in, you are on the wrong machine (or your `~/.login` doesn't set up the CAD environment).

The script is **tcsh**, not bash. Run it with `tcsh` (installed on the RCF machines) or make it executable.

## Usage

1. Copy `get_cad_measurements.tcsh` to the CAD-enabled machine, and make a working directory for the output.
2. In that directory, create `scan_start_end_times.txt` with the start and end of the period you want, in this format (one `Start:` and one `End:` line, `YYYY-MM-DD HH:MM:SS`):

   ```
   Start:	2024-10-16 21:57:23
   End:	2024-10-16 22:27:48
   ```

3. Run it **from that directory**. The times file is read via a relative path, and all output is written to the current directory:

   ```bash
   tcsh get_cad_measurements.tcsh
   ```

4. Copy the resulting `*.dat` files and `profiles/` back to wherever you analyze them.

The script pads the window before querying: the start is moved 1 minute earlier and the end 2 minutes later, both truncated to the minute. Times are interpreted by `date -d` in the machine's local time zone, which is what the logger expects (`MM/DD/YYYY HH:MM:SS`). Use local (BNL) time in the times file.

## Output

| File | Contents |
|---|---|
| `bpms.dat` | IP8 BPM readings (`bip8_bx_mon`, `yip8_bx_mon`) |
| `crossing_angle.dat` | Blue (`BH8CrossingAngle`), yellow (`YH8CrossingAngle`) and blue-minus-yellow (`GH8CrossingAngle`) horizontal crossing angle, computed from the b7/b8 and y7/y8 BPM cells divided by 16250 |
| `bunch_widths.dat` | WCM per-bunch circulating bunch widths, yellow and blue |
| `avg_widths.dat` | WCM average circulating bunch width, yellow and blue |
| `blue_ions.dat`, `yellow_ions.dat` | Total ions from the WCM (`*_TotalIons`) and the DCCT (`beamIons`) |
| `emittance.dat` | IPM emittance |
| `profiles/blue_profile_<yy_HH_MM_SS>.dat`, `profiles/yellow_profile_<...>.dat` | WCM longitudinal (beam profile) data, one 1-second grab every 10 seconds across the window (`bo2-wcm3` blue, `yi2-wcm3` yellow). The suffix is the year, hour, minute and second of the grab. |

The profile loop makes two `exportLoggerData` calls per 10 s, so a 30 minute scan takes a few minutes and produces several hundred files. To change the sampling, edit `interval_seconds` in the script.

## Existing data and plotting

This directory also holds data from the Jul 11 and Aug 12 Vernier scans (`VernierScan_<date>_*.dat`, already collected). `../plot_cad_measurements.py` plots them; set `cad_measurements_path` and `vernier_scan_date` in its `main()` before running.
