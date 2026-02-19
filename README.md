# ECE755-sp26-project

Hardware Accelerator for Reformer LSH Attention

- Paper link: [Reformer: The Efficient Transformer](https://arxiv.org/pdf/2001.04451)
- Blog post: [link](https://research.google/blog/reformer-the-efficient-transformer/)
- Colab: [link](https://colab.research.google.com/github/google/trax/blob/master/trax/models/reformer/image_generation.ipynb)

```
ECE755-sp26-project/  
|--- hardware/                # Physical/Digital Design  
|    |--- design/             # Design specs & architecture diagrams  
|
|--- software/                # Integration Framework  
|    |--- emulation/          # Python functional models of hardware  
|
|--- README.md                # Summary of key objectives, including main contributors  
  
```

### Milestones

**Week 1: Foundations & High-Level Modeling**

- [x] Conduct literature review, specification, and Tiny Tapeout setup.
- [x] Deep-read Reformer paper and LSH theory.
- [ ] Define all design parameters within Tiny Tapeout tile constraints.
- [ ] Set up LibreLane 2 toolchain and Tiny Tapeout template repo.
- [x] Implement Python/NumPy golden reference model.
- [x] Create High-level block diagram/algorithmic simulation.

**Weeks 2–3: Initial RTL & Trial Synthesis**

- [ ] Develop Low-level block diagram/verification.
- [ ] Design LSH Hashing Unit and SPI controller RTL.
- [ ] Design and verify the SPI interface to external SRAM, random-projection MAC array, and argmax bucketing.
- [ ] Develop unit-level testbenches.
- [ ] Ensure Behavioral model is coded & unit test (1–2 sub-blocks) is complete.
- [ ] Run initial synthesis through LibreLane to validate area feasibility.
- [ ] Complete Trial synthesis, ensuring the design flow is debugged.

**Weeks 4–5: Core Processing Logic**

- [ ] Develop Bucket Sort, Chunk Formation, and Chunked Dot-Product Engine RTL.
- [ ] Implement counting-sort with external SRAM scatter/gather.
- [ ] Build serial MAC for chunked QKT computation with score write-back to external SRAM.
- [ ] Verify sorting and dot-product correctness against golden model.

**Weeks 6–7: Integration & End-to-End Verification**

- [ ] Execute top-level integration and end-to-end verification.
- [ ] Integrate controller FSM, SPI memory subsystem, full pipeline across both tiles.
- [ ] Run Cocotb testbench vs. golden model with SPI SRAM behavioral model.
- [ ] Validate functional correctness across sequence lengths.

**Weeks 8–9: Hardening & Submission**

- [ ] Perform LibreLane hardening and Tiny Tapeout submission.
- [ ] Run full LibreLane flow (synthesis, floorplan, placement, CTS, routing, signoff).
- [ ] Execute RTL, synthesis, place & route, annotation.
- [ ] Finalize global design: power, clock, routing.**
- [ ] Fix timing/DRC violations.
- [ ] Generate GDS for Tiny Tapeout submission.
- [ ] Collect area, timing, and power data.
