//
// Created by dn277127 on 1/27/25.
//

#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TH1D.h"
#include <iostream>
using namespace Pythia8;

// Function to check if a particle hits a detector
bool check_hit(double px, double py, double pz, double x, double y, double z,
               double z_detector, double z_thickness, double r_inner, double r_outer) {
    // Compute the z bounds of the detector. Assume center of detector is at z_detector.
    double z_min = z_detector - z_thickness / 2.0;
    double z_max = z_detector + z_thickness / 2.0;

    if (pz == 0) {  // Avoid division by zero, particle is perpendicular to beamline
        if (z < z_min || z > z_max) {
            return false;
        } else {
            return true;
        }
    }

    // Check if particle can reach the detector
    double scale_min = (z_min - z) / pz;  // Scale factor to reach z_min
    double scale_max = (z_max - z) / pz;  // Scale factor to reach z_max

    if (scale_min > scale_max) {
        std::swap(scale_min, scale_max);
    }

    double x_min = x + px * scale_min;
    double y_min = y + py * scale_min;
    double r_min = std::sqrt(x_min * x_min + y_min * y_min);

    double x_max = x + px * scale_max;
    double y_max = y + py * scale_max;
    double r_max = std::sqrt(x_max * x_max + y_max * y_max);

    // If r_min and r_max are both inside or outside the detector, the particle misses. Otherwise, it hits.
    // Assuming straight-line trajectories.
    if (r_min < r_inner && r_max < r_inner) {
        return false;
    }
    if (r_min > r_outer && r_max > r_outer) {
        return false;
    }
    return true;
}

int main() {
    // Initialize Pythia
    Pythia pythia;
    pythia.readString("Beams:idA = 2212");  // Beam A: Proton (PDG ID = 2212)
    pythia.readString("Beams:idB = 2212");  // Beam B: Proton (PDG ID = 2212)
    pythia.readString("Beams:eCM = 510.");  // Center-of-mass energy
    pythia.readString("HardQCD:all = on");  // Enable QCD processes

    // Initialize Pythia
    pythia.init();

    // Detector geometry parameters
    const double z_detector = 250.0;  // Distance along the beamline (cm)
    const double z_thickness = 3.0;  // Thickness of the detector (cm)
    const double r_inner = 5.0;     // Inner radius of annulus (cm)
    const double r_outer = 15.0;     // Outer radius of annulus (cm)
    const double energy_threshold = 0.1;  // Minimum energy to count (GeV)

    // Z-vertex shift parameters
    const double z_shift_min = -250.0;  // Minimum z-vertex shift (cm)
    const double z_shift_max = 250.0;   // Maximum z-vertex shift (cm)
    const double z_shift_step = 1.0;    // Step size for z-vertex shift (cm)

    // Create ROOT file and histogram
    TFile *outputFile = new TFile("detector_hits.root", "RECREATE");
    TH1D *coincident_hits_hist = new TH1D("coincident_hits", "Coincident Hits vs Z-Vertex Shift",
                                          (z_shift_max - z_shift_min) / z_shift_step + 1,
                                          z_shift_min, z_shift_max);

    // Generate events
    const int nEvents = 100000;  // Number of events
    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (!pythia.next()) continue;

        // Loop over z-vertex shifts
        for (double z_shift = z_shift_min; z_shift <= z_shift_max; z_shift += z_shift_step) {
            bool hit_detector_1 = false;
            bool hit_detector_2 = false;

            // Loop over particles in the event
            for (int i = 0; i < pythia.event.size(); ++i) {
                Particle &p = pythia.event[i];

                // Skip initial state particles
                if (!p.isFinal()) continue;

                // Get particle properties
                double px = p.px();
                double py = p.py();
                double pz = p.pz();
                double energy = p.e();
                double x = p.xProd();  // Production vertex x (cm)
                double y = p.yProd();  // Production vertex y (cm)
                double z = p.zProd() + z_shift;  // Shifted z-vertex (cm)

                if (energy < energy_threshold) continue;  // Skip low-energy particles

                // Check hits for both detectors
                if (!hit_detector_1) {
                    hit_detector_1 = check_hit(px, py, pz, x, y, z, z_detector, z_thickness, r_inner, r_outer);
                }
                if (!hit_detector_2) {
                    hit_detector_2 = check_hit(px, py, pz, x, y, z, -z_detector, z_thickness, r_inner, r_outer);
                }

                if (hit_detector_1 && hit_detector_2) {
                    break;  // Already hit, no need to check other particles
                }
            }

            // Increment histogram for coincident hits
            if (hit_detector_1 && hit_detector_2) {
                coincident_hits_hist->Fill(z_shift);
            }
        }
    }

    // Write histogram to ROOT file
    outputFile->cd();
    coincident_hits_hist->Write();
    outputFile->Close();

    // Clean up
    delete outputFile;

    std::cout << "Simulation complete. Results saved to detector_hits.root." << std::endl;

    return 0;
}