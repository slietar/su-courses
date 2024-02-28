load ../drive/FBN1_AlphaFold.pdb, tb5_full
load ../domains/TB5/TB5_cc87b_unrelaxed_rank_001_alphafold2_ptm_model_4_seed_000.pdb, tb5
color orange, %tb5
color green, %tb5_full
color red, %tb5_full and resi 1689-1762
alter %tb5, resi=str(int(resi)+1688)
color marine, (%tb5 or %tb5_full) and resi 1696+1696+1699+1699+1700+1706+1710+1714+1719+1721+1722+1726+1726+1728+1728+1730+1733+1733+1733+1735+1736+1736+1750+1758+1758+1761+1762+1762+1721+1748+1748+1748+1692+1708+1719+1720
align tb5_full, tb5

disable tb5
disable tb5_full
