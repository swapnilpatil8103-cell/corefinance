"""Bank M&A CET1 & Accretion Simulator (Project #2), Stage 3: purchase
accounting for ONE deal between an acquirer and a target, each already
modeled standalone by `corefin.bank` (Stage 2) -- consideration (price-
to-TBV, stock/cash mix), fair value marks (credit mark split PCD/non-PCD
with the Day-2 CECL allowance, loan rate mark, securities mark), CDI,
goodwill, and the pro forma combined balance sheet/CET1 bridge AT CLOSE.

Deal-wide EPS accretion/dilution and TBV earnback over time (Stage 4),
the severely-adverse stress path (Stage 5), and sensitivities (Stage 6)
are not this module's job -- it produces the single jump-off/close-date
snapshot those later stages build on."""
