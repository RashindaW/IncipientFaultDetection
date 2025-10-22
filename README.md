# DyEdgeGAT Refrigeration Dataset Notes

This repository uses the CO₂ supermarket refrigeration benchmark released with the DyEdgeGAT paper.  The full dataset now lives under `Dataset/` with five baseline (normal) CSVs and six fault CSVs.  Each file exposes the same 185 columns, covering command signals, process measurements, and derived KPIs.

## Timestamp
- `Timestamp` – chronological identifier for each sample.  Use it only for ordering/windowing, not as a direct input feature.

## Control / Operating-Condition Variables (6)
These are the thermostat and ambient setpoints that capture the operating condition context described in the paper (control variables + external factors).  They should populate `cfg.dataset.ocvar_dim = 6`.
- `Tsetpt` – case temperature set-point broadcast by the supervisory controller.
- `RHsetpt` – relative humidity target for the display case air.
- `TstatSuc` – suction-side thermostat reference used for compressor staging.
- `TstatCondExit` – condenser exit temperature thermostat reference.
- `TstatDisc` – discharge-line thermostat reference.
- `TstatSubClExit` – sub-cooler exit thermostat reference.

## Measurement Variables
All remaining columns are measurements or derived physical quantities.  They provide the multivariate time-series input to the DyEdgeGAT encoder.

- **Compressor & rack drive commands**  
  `W_MT-COMP1`, `W_MT-COMP2`, `W_MT-COMP3`, `W_LT-COMP1`, `W_LT-COMP2`, `W-CONDENSOR`, `M-MTcooler`, `M-LTcooler`, `M-CompRack`, `W-LT-BPHX`, `W-MT-BPHX`

- **Pressures**  
  `P-LT-BPHX`, `P-MT-BPHX`, `P-MTcase-SUC-inside`, `P-MTcase-SUC`, `P-MTcase-LIQ`, `P-LTcase-SUC-inside`, `P-LTcase-SUC`, `P-LTcase-LIQ`, `P-FlashTank`, `P-GC-IN`, `P-MT_Dis-OilSepIn`, `P-LT-SUC`, `P-MT_SUC`

- **Temperatures (compressors, gas cooler, cases, distributed nodes, heat exchangers)**  
  `T-MT-COMP1-SUC`, `T-MT-COMP1-DIS`, `T-MT-COMP2-SUC`, `T-MT-COMP2-DIS`, `T-MT-COMP3-SUC`, `T-MT-COMP3-DIS`, `T-LT-COMP1-SUC`, `T-LT-COMP1-DIS`, `T-LT-COMP2-SUC`, `T-LT-COMP2-DIS`, `T-GC-SUC`, `T-GC-DIS`, `T-BP-EEVout`, `T-BP-EEVin`, `T-BP-srf`, `T-MTrack-LIQ`, `T-MTcase-LIQ`, `T-spare-13B`, `T-LTcase-LIQ-srf`, `T-LTcase-SUC-srf`, `T-LTcase-EEVout`, `T-LTcase-Sup`, `T-LTcase-Ret`, `T-LTcase-Liq`, `T-LTcase-Suc`, `T-spare-16C`, `T-GC-In`, `T-GC-Out`, `T-GC-Fan2-In`, `T-GC-Fan1-In`, `T-GC-Fan2-Out`, `T-GC-Fan1-Out`, `T-MTRack-Suc-srf`, `T-LT-Ret`, `T-LT-Suc`, `T-LT-Dis`, `T-MT-Suc`, `T-MT-Dis`, `T-FalseLoad`, `T-Spare-2D`, `T-spare-3D`, `T-Spare-4D`, `T-MTCase-Liq-Srf`, `T-MTCase-Suc-Srf`, `T-MTCase-EEVOut`, `T-MTCase-Sup`, `T-MTCase-Ret`, `T-MTCase-Suc`, `T-101`, `T-102`, `T-103`, `T-104`, `T-105`, `T-106`, `T-107`, `T-108`, `T-109`, `T-110`, `T-111`, `T-112`, `T-113`, `T-114`, `T-115`, `T-116`, `T-201`, `T-202`, `T-203`, `T-204`, `T-205`, `T-206`, `T-207`, `T-208`, `T-209`, `T-210`, `T-211`, `T-212`, `T-213`, `T-214`, `T-215`, `T-216`, `T-301`, `T-302`, `T-303`, `T-304`, `T-305`, `T-306`, `T-307`, `T-308`, `T-309`, `T-310`, `T-311`, `T-312`, `T-313`, `T-314`, `T-315`, `T-316`, `T-401`, `T-402`, `T-403`, `T-404`, `T-405`, `T-406`, `T-407`, `T-408`, `T-409`, `T-410`, `T-411`, `T-412`, `T-413`, `T-414`, `T-415`, `T-416`, `T-501`, `T-502`, `T-503`, `T-504`, `T-505`, `T-506`, `T-507`, `T-508`, `T-509`, `T-510`, `T-511`, `T-512`, `T-513`, `T-514`, `T-515`, `T-516`, `T-LT_BPHX_H20_INLET`, `T-LT_BPHX_H20_OUTLET`, `T-MT_BPHX_H20_INLET`, `T-MT_BPHX_H20_OUTLET`, `T-LT_BPHX_C02_EXIT`, `T-MT_BPHX_C02_EXIT`

- **Flows**  
  `F-LT-BPHX`, `F-MT-BPHX`

- **Derived thermodynamic metrics & KPIs**  
  `SupHCompSuc`, `SupHCompDisc`, `SupHEvap1`, `SupHEvap2`, `SubClComCond`, `SubcoolCond1`, `SubcoolCond2`, `SuncoolLiq`, `RefHSct`, `RefHLiq`, `AirHRet`, `AirHSup`, `CapaAirside`, `CapaRefrside`, `EnergyBalance`, `EERA`, `EER`

- **Placeholder column**  
  `Unnamed: 161` – empty column carried over from the logging system; treat it as a measurement feature only if you explicitly impute/clean it.

When constructing the full feature tensor for the model:
1. Use the six control variables above as `operating condition` inputs `U`.
2. Use the remaining 178 measurement variables (excluding `Timestamp`) as the multivariate sequence `X`.
3. Remember to drop or fill `Unnamed: 161` because it is often constant/NaN.

This mapping should make it easier to move from the reduced test subset to the full CO₂ dataset without renaming columns later.

## Multi-GPU Training
- Multi-GPU execution now relies on PyTorch DistributedDataParallel. Launch the scripts with `torchrun --nproc_per_node=<num_gpus>`; each process owns one GPU and runs its own DataLoader pipeline.
- Pick a specific set of GPUs with `CUDA_VISIBLE_DEVICES`. Example (per-GPU batch size of 128, AMP enabled, 4 workers per rank):
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
      train_dyedgegat.py --epochs 20 --batch-size 128 \
      --num-workers 4 --use-amp
  ```
- `--cuda-device` remains for single-GPU selection when you run the script directly (without torchrun). `--cuda-devices` accepts only one id and will otherwise raise.
- `--dist-backend` defaults to `nccl`; switch to `gloo` only when running on CPU.
- The reported batch size is per process. Increase it to fully utilize the additional memory each GPU now owns.
