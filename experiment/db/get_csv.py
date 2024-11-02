import numpy as np
import sqlite3
from datetime import datetime
import pandas as pd
slurm_db_path = "/home1/jongminm/LDPUts/experiment/db/"
con = sqlite3.connect(slurm_db_path+"071424_LDPUts.db")



table_name="conti_location"
dim_vec = [3, 4, 5]
priv_lev_vec = [0.5, 1, 2]
sample_size_multplier = [1000,2700,8400]
for i, d in enumerate(dim_vec):
	for priv_lev in priv_lev_vec:
		cursor = con.execute(
			f"""
			SELECT
				avg(p_val<0.05) as power,
				mechanism||statistic as method,
				sample_size
			FROM {table_name}
			WHERE
				dim = ? and
				priv_lev = ? and
				sample_size
			GROUP BY statistic, mechanism, sample_size
			""",
			[d, priv_lev])
		table = cursor.fetchall()
		table_pd = pd.DataFrame(table)
		table_pd.columns = ["power", "method", "n"]
		table_pd = table_pd[table_pd['n'] % sample_size_multplier[i] == 0]
		table_pd = table_pd.pivot(index= "n", columns = "method", values = "power")
		table_pd.to_csv(slurm_db_path+"location_power_d"+str(d)+ "_priv"+ str(int(10*priv_lev)) +".csv")


table_name="conti_scale"
dim_vec = [3, 4, 5]
priv_lev_vec = [0.5, 1, 2]
sample_size_multplier = [3200,11600,40000]
for i, d in enumerate(dim_vec):
	for priv_lev in priv_lev_vec:
		cursor = con.execute(
			f"""
			SELECT
				avg(p_val<0.05) as power,
				mechanism||statistic as method,
				sample_size
			FROM {table_name}
			WHERE
				dim = ? and
				priv_lev = ? and
				sample_size
			GROUP BY statistic, mechanism, sample_size
			""",
			[d, priv_lev])
		table = cursor.fetchall()
		table_pd = pd.DataFrame(table)
		table_pd.columns = ["power", "method", "n"]
		table_pd = table_pd[table_pd['n'] % sample_size_multplier[i] == 0]
		table_pd = table_pd.pivot(index= "n", columns = "method", values = "power")
		table_pd.to_csv(slurm_db_path+"scale_power_d"+str(d)+ "_priv"+ str(int(10*priv_lev)) +".csv")

table_name="multinomial_perturbunif"
dim_vec = [4, 40, 400]
priv_lev_vec = [0.5, 1, 2]
sample_size_multplier = [2000,3000,20000]
for i, d in enumerate(dim_vec):
	for priv_lev in priv_lev_vec:
		cursor = con.execute(
			f"""
			SELECT
				avg(p_val<0.05) as power,
				mechanism||statistic as method,
				sample_size
			FROM {table_name}
			WHERE
				dim = ? and
				priv_lev = ? and
				sample_size
			GROUP BY statistic, mechanism, sample_size
			""",
			[d, priv_lev])
		table = cursor.fetchall()
		table_pd = pd.DataFrame(table)
		table_pd.columns = ["power", "method", "n"]
		table_pd = table_pd[table_pd['n'] % sample_size_multplier[i] == 0]
		table_pd = table_pd.pivot(index= "n", columns = "method", values = "power")
		table_pd.to_csv(slurm_db_path+"purturbunif_power_d"+str(d)+ "_priv"+ str(int(10*priv_lev)) +".csv")

