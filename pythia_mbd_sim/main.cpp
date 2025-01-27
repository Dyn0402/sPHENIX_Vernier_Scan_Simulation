//
// Created by dn277127 on 1/27/25.
//

#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TH1D.h"
#include <iostream>
using namespace Pythia8;

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
    const double r_inner = 48.8;     // Inner radius of annulus (cm)
    const double r_outer = 51.2;     // Outer radius of annulus (cm)
    const double energy_threshold = 1.0;  // Minimum energy to count (GeV)

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

                // Compute particle trajectory
                double z_target_1 = z_detector - z;  // z displacement to detector 1
                double z_target_2 = -z_detector - z; // z displacement to detector 2

                // Check if particle intersects detector 1 (z = +250 cm)
                if (pz > 0) {
                    double scale = z_target_1 / pz;  // Scale factor to reach z_target_1
                    double x_at_detector = x + px * scale;
                    double y_at_detector = y + py * scale;
                    double r_at_detector = std::sqrt(x_at_detector * x_at_detector + y_at_detector * y_at_detector);

                    if (r_at_detector >= r_inner && r_at_detector <= r_outer && energy > energy_threshold) {
                        hit_detector_1 = true;
                    }
                }

                // Check if particle intersects detector 2 (z = -250 cm)
                if (pz < 0) {
                    double scale = z_target_2 / pz;  // Scale factor to reach z_target_2
                    double x_at_detector = x + px * scale;
                    double y_at_detector = y + py * scale;
                    double r_at_detector = std::sqrt(x_at_detector * x_at_detector + y_at_detector * y_at_detector);

                    if (r_at_detector >= r_inner && r_at_detector <= r_outer && energy > energy_threshold) {
                        hit_detector_2 = true;
                    }
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
