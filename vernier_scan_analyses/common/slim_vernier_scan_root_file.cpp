// slim_tree.C
// To run: root -l -b -q 'slim_tree.cpp'

#include <TFile.h>
#include <TTree.h>
#include <TString.h>
#include <TObjArray.h>
#include <TObjString.h>

void slim_vernier_scan_root_file() {
    // === Open input file ===
    const char* input_path = "/sphenix/u/takakiku/user_takakiku/auau_vernierscan/gl1p_rootfiles/54733.root";
    const char* output_path = "/sphenix/u/dneffsph/gpfs/vernier_scan_root_files/54733_slimmed.root";
    TFile* infile = TFile::Open(input_path, "READ");
    if (!infile || infile->IsZombie()) {
        printf("Error: cannot open input file %s\n", input_path);
        return;
    }

    // === Get tree ===
    TTree* tree = (TTree*)infile->Get("calo_tree");
    if (!tree) {
        printf("Error: cannot find TTree 'calo_tree' in file.\n");
        infile->Close();
        return;
    }

    // === List of branches to keep ===
    std::vector<std::string> branches_to_keep = {
        "BCO",
        "bunch",
        "mbd_zvtx",
        "mbd_SN_trigger",
        "mbd_S_trigger",
        "mbd_N_trigger",
        "mbd_zvtx_trigger",
        "zdc_SN_trigger",
        "zdc_S_trigger",
        "zdc_N_trigger",
        "zdc_raw_count",
        "zdc_live_count",
        "zdc_scaled_count",
        "mbd_raw_count",
        "mbd_live_count",
        "mbd_scaled_count",
        "mbd_N_raw_count",
        "mbd_S_raw_count",
        "mbd_N_live_count",
        "mbd_S_live_count",
        "mbd_N_scaled_count",
        "mbd_S_scaled_count",
        "zdc_N_raw_count",
        "zdc_S_raw_count",
        "zdc_N_live_count",
        "zdc_S_live_count",
        "zdc_N_scaled_count",
        "zdc_S_scaled_count",
        "GL1_clock_count",
        "GL1_live_count",
        "GL1_scaled_count",
        "GL1P_mbd_raw_count",
        "GL1P_mbd_live_count",
        "GL1P_mbd_scaled_count",
        "GL1P_mbd_N_raw_count",
        "GL1P_mbd_N_live_count",
        "GL1P_mbd_N_scaled_count",
        "GL1P_mbd_S_raw_count",
        "GL1P_mbd_S_live_count",
        "GL1P_mbd_S_scaled_count",
        "GL1P_clock_raw_count",
        "GL1P_clock_live_count",
        "GL1P_clock_scaled_count"
    };

    // === Disable all branches ===
    tree->SetBranchStatus("*", 0);

    // === Enable only selected branches ===
    for (const auto& br : branches_to_keep) {
        tree->SetBranchStatus(br.c_str(), 1);
    }

    // === Create output file ===
    TFile* outfile = TFile::Open(output_path, "RECREATE");
    if (!outfile || outfile->IsZombie()) {
        printf("Error: cannot create output file %s\n", output_path);
        infile->Close();
        return;
    }

    // === Clone tree with only selected branches ===
    TTree* slim_tree = tree->CloneTree(0); // Create empty tree with structure
    Long64_t n_entries = tree->GetEntries();

    for (Long64_t i = 0; i < n_entries; ++i) {
        tree->GetEntry(i);
        slim_tree->Fill();
    }

    // === Write and clean up ===
    slim_tree->Write();
    outfile->Close();
    infile->Close();

    printf("Slimmed tree written to: %s\n", output_path);
}
