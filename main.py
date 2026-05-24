# Databricks notebook source
# MAGIC %md
# MAGIC # Technical Project Implementation
# MAGIC
# MAGIC Part 2 of the CA for Big Data Analytics
# MAGIC
# MAGIC Plan of action:
# MAGIC - Data conversion already done before upload pcap -> csv using Wireshark to multiple files containing 1000000 row sample
# MAGIC - load csv then proceed with Data cleaning
# MAGIC - Data preparation and cleaning
# MAGIC - EDA
# MAGIC - Feature Engineering
# MAGIC - Data Normalization and Encoding
# MAGIC - Dimenstionality reduction using SVD
# MAGIC - Machine learning Model Development and Evaluation

# COMMAND ----------

# DBTITLE 1,Cell 1
"""
Part 2 of the CA for Big Data Analytics - Technical Project Implementation 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

# Load all csv files / get all csv files in folder / Wireshark natively export to latin encoding  

# initially i was load oll files in one go
# files = glob.glob("/Workspace/BDA_CA2/*.csv")  
# df = pd.concat([pd.read_csv(f, encoding='latin-1') for f in files], ignore_index=True)

df = pd.read_csv("/Workspace/BDA_CA2/pcap_0-1m.csv", encoding="latin-1")


# lets look at the sample
print(df.head())

# data dimentions
print(df.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data cleaning and preparation
# MAGIC - extract values from info column to enable feature engineering
# MAGIC - Looking only at L3 packets as this is a peering router on L2 frames should be expected
# MAGIC
# MAGIC Below section extracts TCP, UDP and ICMP packets
# MAGIC

# COMMAND ----------

# Protocol Maps

TCP_LIKE = {
    'HTTP', 'SMTP', 'FTP', 'FTP-DATA', 'SSH', 'SSL', 'SSLv2', 'TLS',
    'IMAP', 'POP', 'HTTPS', 'DCERPC', 'NBSS', 'SMB', 'TELNET',
    'IRC', 'RTSP', 'SIP', 'MSNMS', 'CVSPSERVER', 'NNTP', 'RSYNC',
    'PGSQL', 'MySQL', 'TDS', 'RMI', 'TPKT', 'ICAP', 'DISTCC',
    'TCPCL', 'PPTP', 'OpenVPN', 'WHOIS', 'BZR', 'slsk', 'SAMETIME',
    'SAPRFC', 'SAPNI', 'IPDC', 'X11', 'ANSI C12.22', 'TPM',
    'TFP over TCP', 'QUAKE3', 'RTPproxy', 'TIME', 'Chargen',
    'IPSICTL', 'SOCKS', 'Socks', 'NBDS', 'RADIUS', 'R3',
    'POP/IMF', 'SMTP/IMF',
}

UDP_LIKE = {
    'NTP', 'SNMP', 'TFTP', 'SSDP', 'MDNS', 'NBNS', 'STUN',
    'DHCP', 'RIP', 'RTCP', 'RDT', 'eDonkey', 'Gnutella', 'MANOLITO',
    'UDP/XML', 'CIP I/O', 'Portmap', 'H.225.0', 'MPEG TS',
    'GTPv2', 'LISP', 'CIGI', 'CLDAP', 'giFT',
}

DROP_PROTOCOLS = {
    'ESP', 'AH', 'GRE', 'PPP LCP', 'PPP Comp',
    'IPv4', 'IPv6',
    'ISAKMP',
    'WireGuard',
    'Nano Bootstrap', 'AX4000',
    'VNC', 'SCoP',
}

# FUNCTIONS USED TO EXTRACT MORE DETAIL SUCH AS: PORT, PROTOCOL, FLAGS, ETC FROM INFO COLUMN

def parse_tcp(info: str) -> dict:
    result = {
        'src_port': None, 'dst_port': None,
        'well_known_dst': None, 'ephemeral_src': None,
        'flag_syn': 0, 'flag_ack': 0, 'flag_psh': 0,
        'flag_fin': 0, 'flag_rst': 0, 'flag_urg': 0,
        'flag_ecn': 0,       # ECN-Echo
        'flag_cwr': 0,       # Congestion Window Reduced
        'flag_ae':  0,       # Accurate ECN / reserved bit
        'seq': None, 'ack_num': None, 'win': None, 'payload_len': None,
        'tcp_anomaly': None,
        'size_limited': 0, 'retransmission': 0, 'dup_ack': 0,
        'zero_window': 0, 'window_full': 0, 'keep_alive': 0,
        'is_handshake': 0, 'is_syn_ack': 0, 'is_teardown': 0,
    }

    anomaly = re.search(r'\[TCP ([^\]]+)\]', info)
    if anomaly:
        result['tcp_anomaly'] = f"TCP {anomaly.group(1)}"

    # expanded flag pattern to include ECN/CWR/AE/Reserved
    flags_match = re.search(r'\[([A-Z]{2,}(?:,\s*(?:[A-Z]{2,}|Reserved))*)\]', info)
    if flags_match:
        flag_str     = flags_match.group(1)
        active_flags = {f.strip() for f in flag_str.split(',')}
        result['flag_syn'] = int('SYN'      in active_flags)
        result['flag_ack'] = int('ACK'      in active_flags)
        result['flag_psh'] = int('PSH'      in active_flags)
        result['flag_fin'] = int('FIN'      in active_flags)
        result['flag_rst'] = int('RST'      in active_flags)
        result['flag_urg'] = int('URG'      in active_flags)
        result['flag_ecn'] = int('ECN'      in active_flags)
        result['flag_cwr'] = int('CWR'      in active_flags)
        result['flag_ae']  = int('AE'       in active_flags or 'Reserved' in active_flags)

    clean = re.sub(r'\[[^\]]*\]', '', info).strip()
    ports = re.search(r'(\d+)\s*>\s*(\d+)', clean)
    if ports:
        src_p = int(ports.group(1))
        dst_p = int(ports.group(2))
        result['src_port']       = src_p
        result['dst_port']       = dst_p
        result['well_known_dst'] = int(dst_p < 1024)
        result['ephemeral_src']  = int(src_p >= 49152)

    for field, key in [('Seq', 'seq'), ('Ack', 'ack_num'), ('Win', 'win'), ('Len', 'payload_len')]:
        m = re.search(rf'\b{field}=(\d+)', info)
        if m:
            result[key] = int(m.group(1))

    result['is_handshake'] = int(result['flag_syn'] == 1 and result['flag_ack'] == 0)
    result['is_syn_ack']   = int(result['flag_syn'] == 1 and result['flag_ack'] == 1)
    result['is_teardown']  = int(result['flag_fin'] == 1 or  result['flag_rst'] == 1)

    info_l = info.lower()
    result['size_limited']   = int('size limited'   in info_l)
    result['retransmission'] = int('retransmission' in info_l)
    result['dup_ack']        = int('dup ack'        in info_l)
    result['zero_window']    = int('zero window'    in info_l)
    result['window_full']    = int('window full'    in info_l)
    result['keep_alive']     = int('keep-alive'     in info_l)

    return result


def parse_udp(info: str) -> dict:
    result = {
        'src_port': None, 'dst_port': None,
        'payload_len': None,
        'well_known_dst': None, 'ephemeral_src': None,
    }

    clean = re.sub(r'\[[^\]]*\]', '', info).strip()
    ports = re.search(r'(\d+)\s*>\s*(\d+)', clean)
    if ports:
        src_p = int(ports.group(1))
        dst_p = int(ports.group(2))
        result['src_port']       = src_p
        result['dst_port']       = dst_p
        result['well_known_dst'] = int(dst_p < 1024)
        result['ephemeral_src']  = int(src_p >= 49152)

    m = re.search(r'\bLen=(\d+)', info)
    if m:
        result['payload_len'] = int(m.group(1))

    return result


def parse_icmp(info: str) -> dict:
    result = {
        'icmp_type': None,
        'icmp_direction': None,
        'icmp_id': None,
        'icmp_seq': None,
        'icmp_code': None,
    }

    if not isinstance(info, str):
        return result

    info_l = info.lower()

    # icmp_direction — what kind of ICMP message
    if 'echo (ping)' in info_l:
        result['icmp_direction'] = 'echo'
    elif 'destination unreachable' in info_l:
        result['icmp_direction'] = 'unreachable'
    elif 'time-to-live exceeded' in info_l:
        result['icmp_direction'] = 'ttl_exceeded'
    elif 'redirect' in info_l:
        result['icmp_direction'] = 'redirect'

    # icmp_type — request or reply
    if 'request' in info_l:
        result['icmp_type'] = 'request'
    elif 'reply' in info_l:
        result['icmp_type'] = 'reply'

    # icmp_code — text description inside parentheses
    # e.g. "Destination unreachable (Port unreachable)" -> "port unreachable"
    code_match = re.search(r'\(([^)]+)\)', info)
    if code_match:
        result['icmp_code'] = code_match.group(1).strip().lower()

    # icmp_id
    icmp_id = re.search(r'id=0x([0-9a-fA-F]+)', info)
    if icmp_id:
        result['icmp_id'] = int(icmp_id.group(1), 16)

    # icmp_seq — first number before the slash e.g. seq=62669/52724
    icmp_seq = re.search(r'seq=(\d+)', info)
    if icmp_seq:
        result['icmp_seq'] = int(icmp_seq.group(1))

    return result


def parse_info(info, protocol: str = None) -> dict:
    if not isinstance(info, str):
        return {}

    proto = (protocol or '').strip().upper()

    if proto in {p.upper() for p in DROP_PROTOCOLS}:
        return {}
    elif proto in {p.upper() for p in TCP_LIKE}:
        return parse_tcp(info)
    elif proto in {p.upper() for p in UDP_LIKE}:
        return parse_udp(info)
    elif proto == 'DNS':
        if re.search(r'\b(SYN|ACK|FIN|RST)\b', info):
            return parse_tcp(info)
        else:
            return parse_udp(info)
    elif proto == 'TCP':
        return parse_tcp(info)
    elif proto == 'UDP':
        return parse_udp(info)
    elif proto in ('ICMP', 'ICMP, HIPERCONTRACER'):
        return parse_icmp(info)
    else:
        if re.search(r'\b(SYN|ACK|FIN|RST)\b', info):
            return parse_tcp(info)
        elif re.search(r'\d+\s*>\s*\d+', info):
            return parse_udp(info)
        else:
            return {}


def assign_protocol_group(proto: str) -> str:
    p = (proto or '').strip().upper()
    if p == 'TCP' or p in {x.upper() for x in TCP_LIKE}:
        return 'TCP'
    elif p == 'UDP' or p in {x.upper() for x in UDP_LIKE}:
        return 'UDP'
    elif p in ('ICMP', 'ICMP, HIPERCONTRACER'):
        return 'ICMP'
    elif p in {x.upper() for x in DROP_PROTOCOLS}:
        return 'DROP'
    else:
        return 'OTHER'


def expand_info_column(df: pd.DataFrame,
                       info_col: str = 'Info',
                       protocol_col: str = 'Protocol') -> pd.DataFrame:
    parsed = df.apply(
        lambda row: parse_info(row[info_col], row.get(protocol_col)),
        axis=1
    )
    parsed_df = pd.json_normalize(parsed)
    df_out = pd.concat([df.reset_index(drop=True), parsed_df], axis=1)
    # assign protocol group here so it's always done together
    df_out['protocol_group'] = df_out[protocol_col].apply(assign_protocol_group)
    return df_out

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Fields extracted per protocol
# MAGIC
# MAGIC - TCP: ports, 6 flags, seq/ack/win/payload length, TCP anomaly string, 6 flow signal flags, 3 engineered flag combinations, port range indicators
# MAGIC - UDP: ports, payload length, port range indicators
# MAGIC - ICMP: type, direction, id, seq, code
# MAGIC
# MAGIC ### Promlem found
# MAGIC was that Wireshark was by default labaleing TCP protocols like HTTP as a separate protocol not TCP example ouput:
# MAGIC
# MAGIC **Protocol
# MAGIC TCP     43615
# MAGIC UDP      2046
# MAGIC HTTP     1307
# MAGIC DNS       722
# MAGIC SMTP      419**
# MAGIC
# MAGIC ### Solution: Protocol 
# MAGIC
# MAGIC Map groups aplication protocols in to TCP / UDP
# MAGIC

# COMMAND ----------


df_expanded = expand_info_column(df)

# Show Validation

print("Shape:", df_expanded.shape)
print()
print("Protocol groups:")
print(df_expanded['protocol_group'].value_counts())
print()

cols_to_check = ['src_port', 'dst_port', 'flag_syn', 'seq', 'payload_len', 'icmp_type']
cols_present  = [c for c in cols_to_check if c in df_expanded.columns]
print("Null counts:")
print(df_expanded[cols_present].isnull().sum())
print()

# Where are ports still null
null_ports = df_expanded[df_expanded['src_port'].isnull()]
print("Protocols with null ports:")
print(null_ports['Protocol'].value_counts().head(20))
print()

# What do those Info strings look like
print("Sample Info for null-port rows:")
print(null_ports[['Protocol', 'Info']].drop_duplicates('Protocol').to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC Summary
# MAGIC
# MAGIC Beside the information that I am intrested in above section also clasify protocols as Others that are not define/mixed some have ports some dont. Drop group represent for instance malformes, encrypted or tunneled traffic i.e.: IPSec, PPP, ESP, AH etc... Both categories remain in the dataset but as per bellow stats:
# MAGIC
# MAGIC - DROP  = 5,519 rows  (0.55%) 
# MAGIC - OTHER = 15,671 rows (1.57%) 
# MAGIC
# MAGIC Remaining protocols in numbers:
# MAGIC
# MAGIC - TCP Rows: 922,4719 - 4.6%
# MAGIC - UDP Rows: 51,699 - 5.3%
# MAGIC - ICMP Rows: 4,640 - 0.5%
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Cleaning  
# MAGIC
# MAGIC Getting the basline first:

# COMMAND ----------


print(f"Total rows:    {len(df_expanded)}")
print(f"Total columns: {len(df_expanded.columns)}")

print("Protocol groups:")
print(df_expanded['protocol_group'].value_counts())

print("Null counts across key columns:")
print(df_expanded[['src_port','dst_port','flag_syn','seq','payload_len','icmp_type']].isnull().sum())

print("Duplicate rows (same src, dst, protocol, length, info):")
dupes = df_expanded.duplicated(subset=['Source','Destination','Protocol','Length','Info'])
print(f"  {dupes.sum()} duplicates found")

# COMMAND ----------

# MAGIC %md
# MAGIC ### SUMMARY
# MAGIC
# MAGIC - Originally 7 colums, now 37
# MAGIC
# MAGIC - what is intersting in above is that I am seeing: 83042 duplicates although this is large dataset duplicates stand for 8.30% of entire dataset, this could be the result of same data passing multiple interfaces on the same device (peering router in this case) but also fack that the dataset was anonymized and no paylod is present, detecting duplicates is harder
# MAGIC
# MAGIC - 83,474 null ports — this is a mix of ICMP
# MAGIC
# MAGIC - 997,370 null icmp_type — expected, everything that isn't ICMP will be null here.
# MAGIC
# MAGIC What makes a true duplicate?
# MAGIC Same source, destination, protocol, length and info = same packet captured twice
# MAGIC - keep the first occurrence and drop the rest
# MAGIC
# MAGIC
# MAGIC ### Dropping data 
# MAGIC
# MAGIC - removing groups DROP and OTHER and duplicates
# MAGIC

# COMMAND ----------

# Keep only TCP, UDP, ICMP
df_clean = df_expanded[
    df_expanded['protocol_group'].isin(['TCP', 'UDP', 'ICMP'])
].copy()

# After
print(f"Rows after:  {len(df_clean)}")
print(f"Rows removed: {len(df_expanded) - len(df_clean)}")
print()
print("Remaining protocol groups:")
print(df_clean['protocol_group'].value_counts())

# COMMAND ----------

# Remove Duplicates

rows_before = len(df_clean)
print(f"Rows before: {rows_before}")

df_clean = df_clean.drop_duplicates(
    subset=['Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info']
).copy()

rows_after = len(df_clean)
print(f"Rows after:  {rows_after}")
print(f"Duplicates removed: {rows_before - rows_after}")
print()
print("Protocol groups after dedup:")
print(df_clean['protocol_group'].value_counts())

# Investigate UDP dedup impact

udp_before = df_expanded[df_expanded['protocol_group'] == 'UDP']['Protocol'].value_counts()
udp_after  = df_clean[df_clean['protocol_group'] == 'UDP']['Protocol'].value_counts()

udp_comparison = pd.DataFrame({
    'before': udp_before,
    'after':  udp_after
}).fillna(0).astype(int)

udp_comparison['removed']     = udp_comparison['before'] - udp_comparison['after']
udp_comparison['pct_removed'] = (udp_comparison['removed'] / udp_comparison['before'] * 100).round(1)

print(udp_comparison.sort_values('removed', ascending=False).head(15).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 190 true duplicates — packets captured at the exact same timestamp with identical headers. That's 0.019% of the dataset which make sense

# COMMAND ----------

# MAGIC %md
# MAGIC Next - Null Flag columns

# COMMAND ----------

# Fill Null Flag Columns

flag_cols = [
    'flag_syn', 'flag_ack', 'flag_psh', 'flag_fin', 'flag_rst', 'flag_urg',
    'is_handshake', 'is_syn_ack', 'is_teardown',
    'size_limited', 'retransmission', 'dup_ack',
    'zero_window', 'window_full', 'keep_alive'
]

print("Null counts BEFORE fill:")
print(df_clean[flag_cols].isnull().sum())
print()

df_clean[flag_cols] = df_clean[flag_cols].fillna(0).astype(int)

print("Null counts AFTER fill:")
print(df_clean[flag_cols].isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC 56,322 —  UDP + ICMP rows (51,669 + 4,640 = 56,309, the small difference being a few rows that had no flags parsed). These protocols don't have TCP flags so null correctly became 0
# MAGIC
# MAGIC Next. Validate port ranges (0-65535)
# MAGIC
# MAGIC ICMP rows will have null ports — that is expected and correct, anything outside this range means something went wrong during parsing
# MAGIC
# MAGIC

# COMMAND ----------

# Validate Port Ranges

print("Port range check:")
print(f"  src_port min: {df_clean['src_port'].min()}   max: {df_clean['src_port'].max()}")
print(f"  dst_port min: {df_clean['dst_port'].min()}   max: {df_clean['dst_port'].max()}")
print()

# find invalid ports
invalid = df_clean[
    (df_clean['src_port'].notna() & ((df_clean['src_port'] < 0) | (df_clean['src_port'] > 65535))) |
    (df_clean['dst_port'].notna() & ((df_clean['dst_port'] < 0) | (df_clean['dst_port'] > 65535)))
]
print(f"Rows with invalid port values: {len(invalid)}")

if len(invalid) > 0:
    print(invalid[['Protocol', 'src_port', 'dst_port', 'Info']].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Summary: no invalid ports

# COMMAND ----------

# Data Summary

print(" NULL SUMMARY ")
print()

# check all columns
null_counts = df_clean.isnull().sum()
null_pct    = (null_counts / len(df_clean) * 100).round(2)

null_summary = pd.DataFrame({
    'null_count': null_counts,
    'null_pct':   null_pct
})

# only show columns that have nulls
null_summary = null_summary[null_summary['null_count'] > 0].sort_values('null_count', ascending=False)

print(null_summary.to_string())
print()

# final shape
print(f" FINAL SHAPE ")
print(f"Rows:    {len(df_clean)}")
print(f"Columns: {len(df_clean.columns)}")
print()

# protocol breakdown
print(" PROTOCOL GROUPS ")
print(df_clean['protocol_group'].value_counts())
print()

# compare to original
print(" CLEANING SUMMARY ")
print(f"Original rows:  1,000,000")
print(f"Final rows:     {len(df_clean)}")
print(f"Rows removed:   {1_000_000 - len(df_clean)} ({((1_000_000 - len(df_clean)) / 1_000_000 * 100):.2f}%)")

# COMMAND ----------

print(df_clean[df_clean['protocol_group'] == 'ICMP']['Info'].head(20).to_string())

# COMMAND ----------

print(df_clean[df_clean['Info'].isnull()][['Protocol','Source','Destination','Length','Info']])

# COMMAND ----------

# Cleaning Step 7 - Drop null Info rows

rows_before = len(df_clean)
df_clean = df_clean[df_clean['Info'].notna()].copy()

print(f"Rows removed: {rows_before - len(df_clean)}")
print(f"Final rows:   {len(df_clean)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaning Summary
# MAGIC
# MAGIC | Step   | Action                                | Rows Removed | % of Original |
# MAGIC |--------|---------------------------------------|-------------|---------------|
# MAGIC | Step 2 | Dropped DROP and OTHER protocols      | 21,190      | 2.12%         |
# MAGIC | Step 3 | Removed duplicates (Time-based)       | 190         | 0.02%         |
# MAGIC | Step 7 | Dropped null Info rows                | 13          | 0.00%         |
# MAGIC | **Total** |                                    | **21,393**  | **2.14%**     |
# MAGIC
# MAGIC ## Final Dataset
# MAGIC
# MAGIC | Metric          | Value     | Detail                  |
# MAGIC |-----------------|-----------|-------------------------|
# MAGIC | Total Rows      | 978,607   |                         |
# MAGIC | Total Columns   | 37        |                         |
# MAGIC | TCP             | 922,311   | 94.2% of clean data     |
# MAGIC | UDP             | 51,669    | 5.3% of clean data      |
# MAGIC | ICMP            | 4,640     | 0.5% of clean data      |
# MAGIC | Data Retained   | 97.86%    | of original 1,000,000   |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratoty Data Analysis(EDA)
# MAGIC
# MAGIC 1. Decriptive statistics (Mean, Median, Standard deviation, quartiles)
# MAGIC
# MAGIC

# COMMAND ----------

import matplotlib.pyplot as plt

print(" DESCRIPTIVE STATISTICS — ALL NUMERICAL FEATURES ")
print()

num_cols = [
    'Length', 'src_port', 'dst_port', 'seq', 'ack_num',
    'win', 'payload_len', 'icmp_id', 'icmp_seq'
]

stats = df_clean[num_cols].agg([
    'count', 'mean', 'median', 'std',
    lambda x: x.quantile(0.25),
    lambda x: x.quantile(0.75),
    'min', 'max'
])
stats.index = ['count', 'mean', 'median', 'std', 'Q1 (25%)', 'Q3 (75%)', 'min', 'max']
print(stats.round(2).to_string())
print()

# Per protocol  
print(" DESCRIPTIVE STATISTICS ")
print()

protocol_cols = {
    'TCP':  ['Length', 'src_port', 'dst_port', 'seq', 'ack_num', 'win', 'payload_len'],
    'UDP':  ['Length', 'src_port', 'dst_port', 'payload_len'],
    'ICMP': ['Length', 'icmp_id', 'icmp_seq'],
}

for group, cols in protocol_cols.items():
    print(f"--- {group} ---")
    subset = df_clean[df_clean['protocol_group'] == group]
    stats_group = subset[cols].agg([
        'count', 'mean', 'median', 'std',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
        'min', 'max'
    ])
    stats_group.index = ['count', 'mean', 'median', 'std', 'Q1 (25%)', 'Q3 (75%)', 'min', 'max']
    print(stats_group.round(2).to_string())
    print()

# TCP Flag frequency 
print(" TCP FLAG FREQUENCY ")
print()

tcp = df_clean[df_clean['protocol_group'] == 'TCP']
flag_cols = ['flag_syn', 'flag_ack', 'flag_psh', 'flag_fin', 'flag_rst', 'flag_urg']

flag_stats = pd.DataFrame({
    'count': tcp[flag_cols].sum(),
    'pct':   (tcp[flag_cols].sum() / len(tcp) * 100).round(2)
}).sort_values('count', ascending=False)

print(flag_stats.to_string())
print()


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Descriptive Statistics — Key Features', fontsize=14)

# Packet length distribution 
axes[0,0].hist(df_clean['Length'], bins=50, color='steelblue', edgecolor='white')
axes[0,0].axvline(df_clean['Length'].mean(),   color='red',    linestyle='--', label=f"Mean {df_clean['Length'].mean():.0f}")
axes[0,0].axvline(df_clean['Length'].median(), color='orange', linestyle='--', label=f"Median {df_clean['Length'].median():.0f}")
axes[0,0].set_title('Packet Length Distribution')
axes[0,0].set_xlabel('Length (bytes)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# Packet length by protocol 
df_clean.boxplot(column='Length', by='protocol_group', ax=axes[0,1])
axes[0,1].set_title('Packet Length by Protocol')
axes[0,1].set_xlabel('Protocol Group')
axes[0,1].set_ylabel('Length (bytes)')
plt.sca(axes[0,1])
plt.title('Packet Length by Protocol')

# TCP flag frequency bar chart
flag_counts = df_clean[flag_cols].sum()
flag_counts.plot(kind='bar', ax=axes[0,2], color='steelblue', edgecolor='white')
axes[0,2].set_title('TCP Flag Frequency')
axes[0,2].set_xlabel('Flag')
axes[0,2].set_ylabel('Count')
axes[0,2].tick_params(axis='x', rotation=45)

# Source port distribution (log scale)
df_clean['src_port'].dropna().plot(
    kind='hist', bins=100, ax=axes[1,0], color='steelblue', edgecolor='white', logy=True
)
axes[1,0].set_title('Source Port Distribution ')
axes[1,0].set_xlabel('Source Port')
axes[1,0].set_ylabel('Frequency (log)')

# Destination port distribution (log scale)
df_clean['dst_port'].dropna().plot(
    kind='hist', bins=100, ax=axes[1,1], color='orange', edgecolor='white', logy=True
)
axes[1,1].set_title('Destination Port Distribution (log scale)')
axes[1,1].set_xlabel('Destination Port')
axes[1,1].set_ylabel('Frequency (log)')

# Payload length distribution
df_clean['payload_len'].dropna().plot(
    kind='hist', bins=50, ax=axes[1,2], color='green', edgecolor='white', logy=True
)
axes[1,2].set_title('Payload Length Distribution (log scale)')
axes[1,2].set_xlabel('Payload Length (bytes)')
axes[1,2].set_ylabel('Frequency (log)')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Protocol and Packet size distribution

# COMMAND ----------

# Packet Size Distribution

print(" PACKET SIZE CATEGORIES ")
print()

bins   = [0, 128, 512, 1504, float('inf')]
labels = ['Small (<128)', 'Medium (128-512)', 'Large (512-1504)', 'Jumbo (>1504)']

df_clean['size_category'] = pd.cut(df_clean['Length'], bins=bins, labels=labels)

size_dist = df_clean['size_category'].value_counts().sort_index()
size_pct  = (size_dist / len(df_clean) * 100).round(2)

size_summary = pd.DataFrame({
    'count': size_dist,
    'pct':   size_pct
})
print(size_summary.to_string())
print()

# per protocol group
print(" PACKET SIZE CATEGORIES BY PROTOCOL GROUP ")
print()
size_by_proto = df_clean.groupby(['protocol_group', 'size_category'], observed=False).size().unstack(fill_value=0)
size_by_proto_pct = size_by_proto.div(size_by_proto.sum(axis=1), axis=0).mul(100).round(2)
print(size_by_proto_pct.to_string())
print()

# Protocol Distribution

print(" PROTOCOL GROUP DISTRIBUTION ")
print()
proto_dist = df_clean['protocol_group'].value_counts()
proto_pct  = (proto_dist / len(df_clean) * 100).round(2)
proto_summary = pd.DataFrame({'count': proto_dist, 'pct': proto_pct})
print(proto_summary.to_string())
print()

# top 10 protocols within each group
print(" TOP 10 PROTOCOLS WITHIN EACH GROUP ")
print()
for group in ['TCP', 'UDP', 'ICMP']:
    subset = df_clean[df_clean['protocol_group'] == group]
    top    = subset['Protocol'].value_counts().head(10)
    pct    = (top / len(subset) * 100).round(2)
    print(f"--- {group} ({len(subset):,} packets) ---")
    print(pd.DataFrame({'count': top, 'pct': pct}).to_string())
    print()

# Charts 
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Packet Size and Protocol Distributions', fontsize=14)

# Packet size distribution with mean and median
axes[0,0].hist(df_clean['Length'], bins=100, color='steelblue', edgecolor='white')
axes[0,0].axvline(df_clean['Length'].mean(),   color='red',    linestyle='--', linewidth=1.5, label=f"Mean {df_clean['Length'].mean():.0f}b")
axes[0,0].axvline(df_clean['Length'].median(), color='orange', linestyle='--', linewidth=1.5, label=f"Median {df_clean['Length'].median():.0f}b")
axes[0,0].set_title('Packet Size Distribution')
axes[0,0].set_xlabel('Length (bytes)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# Packet size distribution log scale to see full range
axes[0,1].hist(df_clean['Length'], bins=100, color='steelblue', edgecolor='white', log=True)
axes[0,1].axvline(df_clean['Length'].mean(),   color='red',    linestyle='--', linewidth=1.5, label=f"Mean {df_clean['Length'].mean():.0f}b")
axes[0,1].axvline(df_clean['Length'].median(), color='orange', linestyle='--', linewidth=1.5, label=f"Median {df_clean['Length'].median():.0f}b")
axes[0,1].set_title('Packet Size Distribution (log scale)')
axes[0,1].set_xlabel('Length (bytes)')
axes[0,1].set_ylabel('Frequency (log)')
axes[0,1].legend()

# Packet size category pie chart
size_dist.plot(
    kind='pie', ax=axes[0,2], autopct='%1.1f%%',
    colors=['steelblue', 'orange', 'green', 'red'],
    startangle=90
)
axes[0,2].set_title('Packet Size Categories')
axes[0,2].set_ylabel('')

# Protocol group distribution bar chart
proto_dist.plot(kind='bar', ax=axes[1,0], color=['steelblue', 'orange', 'green'], edgecolor='white')
axes[1,0].set_title('Protocol Group Distribution')
axes[1,0].set_xlabel('Protocol Group')
axes[1,0].set_ylabel('Packet Count')
axes[1,0].tick_params(axis='x', rotation=0)
for i, v in enumerate(proto_dist):
    axes[1,0].text(i, v + 1000, f'{v:,}', ha='center', fontsize=9)

# Top 10 TCP protocols
top_tcp = df_clean[df_clean['protocol_group'] == 'TCP']['Protocol'].value_counts().head(10)
top_tcp.plot(kind='barh', ax=axes[1,1], color='orange', edgecolor='white')
axes[1,1].set_title('Top 10 TCP Protocols')
axes[1,1].set_xlabel('Packet Count')
axes[1,1].invert_yaxis()

# Top 10 protocols overall
top_protocols = df_clean['Protocol'].value_counts().head(10)
top_protocols.plot(kind='barh', ax=axes[1,2], color='steelblue', edgecolor='white')
axes[1,2].set_title('Top 10 Protocols by Packet Count')
axes[1,2].set_xlabel('Packet Count')
axes[1,2].invert_yaxis()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------


# Time Column Check
print(" TIME COLUMN ")
print(df_clean['Time'].describe())
print()
print(f"Total capture duration: {df_clean['Time'].max() - df_clean['Time'].min():.2f} seconds")
print(f"Total capture duration: {(df_clean['Time'].max() - df_clean['Time'].min()) / 60:.2f} minutes")
print()

# check if Time is sorted
is_sorted = df_clean['Time'].is_monotonic_increasing
print(f"Timestamps sorted:      {is_sorted}")
print()

# sample
print("First 5 timestamps:")
print(df_clean['Time'].head().to_string())
print()
print("Last 5 timestamps:")
print(df_clean['Time'].tail().to_string())


# COMMAND ----------

# MAGIC %md
# MAGIC Note:
# MAGIC
# MAGIC  only 13.29 seconds of capture data
# MAGIC  thats a very short window, implications for flow duration analysis:
# MAGIC
# MAGIC 978,607 packets in 13.29 seconds = 73,634 packets per second — this is a high speed OC48 backbone link, expected for a peering point
# MAGIC Flow duration will be very short
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Base line for normal traffic:
# MAGIC - valid TCP handshake is always SYN -> SYN+ACK -> ACK
# MAGIC - valid packet is always between 32 and 1504 bytes
# MAGIC - Port 80 traffic should look like HTTP
# MAGIC - single host shouldn't contact thousands of destinations in 13 seconds

# COMMAND ----------

# Baseline checks for normal traffic

# 1. Valid TCP handshake: SYN -> SYN+ACK -> ACK
tcp_handshakes = df_clean[
    (df_clean['protocol_group'] == 'TCP') &
    (df_clean['flag_syn'] == 1)
]
handshake_counts = tcp_handshakes.groupby(['Source', 'Destination']).size()

# 2. Valid packet size: 32-1504 bytes
valid_packet_size = df_clean[
    (df_clean['Length'] >= 32) & (df_clean['Length'] <= 1504)
]

# 3. Port 80 traffic should look like HTTP
port80_http = df_clean[
    (df_clean['dst_port'] == 80) & (df_clean['Protocol'].str.upper() == 'HTTP')
]

# 4. Single host shouldn't contact thousands of destinations in full capture duration
host_dest_counts = df_clean.groupby('Source')['Destination'].nunique()
suspicious_hosts = host_dest_counts[host_dest_counts > 1000]

# 5. IQR-based statistical thresholds for Length, window size, payload
iqr_thresholds = {}
for col in ['Length', 'win', 'payload_len']:
    Q1  = tcp[col].quantile(0.25)
    Q3  = tcp[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = max(Q1 - 1.5 * IQR, 0)  # clamp to 0 — negative sizes impossible
    upper = Q3 + 1.5 * IQR
    iqr_thresholds[col] = {
        'Q1':    round(Q1, 2),
        'Q3':    round(Q3, 2),
        'IQR':   round(IQR, 2),
        'lower': round(lower, 2),
        'upper': round(upper, 2)
    }

# 6. TCP flag combination profiling with readable labels
def flag_combo_label(row):
    flags = ['SYN', 'ACK', 'PSH', 'FIN', 'RST', 'URG']
    return '+'.join([f for f, v in zip(flags, row) if v == 1]) or 'NONE'

flag_cols = ['flag_syn', 'flag_ack', 'flag_psh', 'flag_fin', 'flag_rst', 'flag_urg']
tcp_flag_combos = tcp[flag_cols].apply(flag_combo_label, axis=1)
flag_combo_counts = tcp_flag_combos.value_counts()
normal_flag_combos = flag_combo_counts.head(10)

# 7. Traffic volume baseline — packets/sec, bytes/sec
window = df_clean['Time'].max() - df_clean['Time'].min()
duration = window
packets_per_sec = len(df_clean) / duration
bytes_per_sec = df_clean['Length'].sum() / duration
traffic_volume = {
    'packets_per_sec': packets_per_sec,
    'bytes_per_sec': bytes_per_sec
}

print("TCP handshake SYN packets per Source-Destination pair:")
print(handshake_counts.head())
print("\nValid packet size count:", len(valid_packet_size))
print("\nPort 80 HTTP packet count:", len(port80_http))
print("\nSuspicious hosts contacting >1000 destinations in full capture duration:")
print(suspicious_hosts)
print("\nIQR-based thresholds for Length, win, payload_len:")
print(iqr_thresholds)
print("\nTop 10 TCP flag combinations (normal):")
print(normal_flag_combos)
print("\nTraffic volume baseline (packets/sec, bytes/sec):")
print(traffic_volume)

# COMMAND ----------

# MAGIC %md
# MAGIC This is my first anomally:
# MAGIC
# MAGIC uspicious hosts contacting >1000 destinations in full capture duration:
# MAGIC Source
# MAGIC 3.150.43.198    1596
# MAGIC
# MAGIC

# COMMAND ----------


# Investigate Anomaly Candidates

flag_cols = ['flag_syn', 'flag_ack', 'flag_psh', 'flag_fin', 'flag_rst', 'flag_urg']

# Suspicious host
print(" SUSPICIOUS HOST: 3.150.43.198 ")
suspect = df_clean[df_clean['Source'] == '3.150.43.198']
print(f"Total packets:       {len(suspect):,}")
print(f"Unique destinations: {suspect['Destination'].nunique():,}")
print(f"Unique dst ports:    {suspect['dst_port'].nunique():,}")
print(f"Protocols used:      {suspect['Protocol'].unique()}")
print()
print("Flag breakdown:")
print(suspect[flag_cols].sum().to_string())
print()
print("Top 10 destination ports:")
print(suspect['dst_port'].value_counts().head(10).to_string())
print()

# NONE flag packets 
print(" NONE FLAG PACKETS (no flags set) ")
tcp       = df_clean[df_clean['protocol_group'] == 'TCP']
none_flags = tcp[tcp[flag_cols].sum(axis=1) == 0]
print(f"Total NONE flag packets: {len(none_flags):,}")
print()
print("Top 10 source IPs:")
print(none_flags['Source'].value_counts().head(10).to_string())
print()
print("Top 10 destination ports:")
print(none_flags['dst_port'].value_counts().head(10).to_string())
print()
print("Sample rows:")
print(none_flags[['Source', 'Destination', 'src_port', 'dst_port', 'Length', 'seq', 'Info']].head(10).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Suspicious Host 3.150.43.198 — port Scanner for FTP
# MAGIC
# MAGIC 2,526 packets, 1,596 unique destinations
# MAGIC 1 unique destination port — port 21 (FTP)
# MAGIC 98.8% SYN packets (2,496 out of 2,526) with almost no ACKs
# MAGIC FTP port scan — sending SYN packets to port 21 across thousands of hosts looking for open FTP servers
# MAGIC 29 ACKs means almost none of the targets responded
# MAGIC
# MAGIC NONE Flag Packets — Not a NULL Scan
# MAGIC - NaN for src_port, dst_port, seq — nothing was parseable
# MAGIC - Info field: Continuation [Packet size limited during capture]
# MAGIC - These are not malicious NULL scans — they're parsing artefacts from truncated packets
# MAGIC
# MAGIC

# COMMAND ----------


# Remark NONE Flag Packets

print(" NONE FLAG PACKETS — ROOT CAUSE ")
print()

none_flags = tcp[tcp[flag_cols].sum(axis=1) == 0]

# check how many are size limited
size_limited = none_flags[none_flags['Info'].str.contains('size limited', case=False, na=False)]
continuation = none_flags[none_flags['Info'].str.contains('Continuation', case=False, na=False)]
null_ports   = none_flags[none_flags['src_port'].isna()]

print(f"Total NONE flag packets:         {len(none_flags):,}")
print(f"  Size limited during capture:   {len(size_limited):,} ({len(size_limited)/len(none_flags)*100:.1f}%)")
print(f"  Continuation packets:          {len(continuation):,} ({len(continuation)/len(none_flags)*100:.1f}%)")
print(f"  Null ports (unparseable):      {len(null_ports):,} ({len(null_ports)/len(none_flags)*100:.1f}%)")
print()

# remaining NONE flag packets that are NOT truncated
genuine_none = none_flags[
    ~none_flags['Info'].str.contains('size limited|Continuation', case=False, na=False) &
    none_flags['src_port'].notna()
]
print(f"Genuine NONE flag packets:       {len(genuine_none):,}")
if len(genuine_none) > 0:
    print()
    print("Sample genuine NONE flag packets:")
    print(genuine_none[['Source', 'Destination', 'src_port', 'dst_port', 'Length', 'Info']].head(10).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC 8 outlier types — size, window, zero window, suspicious flags, SYN scan, destination diversity, RST flood and malformed packets

# COMMAND ----------

# Outlier Detection

# Packet size outliers 
df_clean['outlier_size'] = (
    (df_clean['Length'] < 32) | (df_clean['Length'] > 1504)
).astype(int)

# Window size outliers  for TCP only 
win_upper = iqr_thresholds['win']['upper']
win_lower = iqr_thresholds['win']['lower']

df_clean['outlier_win'] = 0
tcp_mask = df_clean['protocol_group'] == 'TCP'
df_clean.loc[tcp_mask, 'outlier_win'] = (
    (df_clean.loc[tcp_mask, 'win'] < win_lower) |
    (df_clean.loc[tcp_mask, 'win'] > win_upper)
).astype(int)

# Zero window - DoS signal
df_clean['outlier_zero_win'] = 0
df_clean.loc[tcp_mask, 'outlier_zero_win'] = (
    df_clean.loc[tcp_mask, 'win'] == 0
).astype(int)

# Suspicious flag combinations 
# SYN+FIN — used stealth scans
# URG — generally not used legitimately
# AE/Reserved — non-standard bits
df_clean['outlier_flags'] = 0
df_clean.loc[tcp_mask, 'outlier_flags'] = (
    # SYN+FIN combination — never valid
    ((df_clean.loc[tcp_mask, 'flag_syn'] == 1) & (df_clean.loc[tcp_mask, 'flag_fin'] == 1)) |
    # URG flag — 2 packets in entire dataset
    (df_clean.loc[tcp_mask, 'flag_urg'] == 1) |
    # AE/Reserved bits set
    (df_clean.loc[tcp_mask, 'flag_ae'] == 1)
).astype(int)

# FTP scanner — high SYN rate to single port 
# flag any source IP sending SYNs to more than 99th percentile unique destinations
syn_packets    = df_clean[(df_clean['protocol_group'] == 'TCP') & (df_clean['flag_syn'] == 1)]
syn_dst_counts = syn_packets.groupby('Source')['Destination'].nunique()
syn_threshold  = syn_dst_counts.quantile(0.99)

high_syn_sources = syn_dst_counts[syn_dst_counts > syn_threshold].index
df_clean['outlier_syn_scan'] = df_clean['Source'].isin(high_syn_sources).astype(int)

# Destination diversity outliers - flag source IPs contacting more destinations than 99th percentile
dst_threshold    = host_dest_counts.quantile(0.99)
high_dst_sources = host_dest_counts[host_dest_counts > dst_threshold].index
df_clean['outlier_dst_diversity'] = df_clean['Source'].isin(high_dst_sources).astype(int)

# RST flood detection - flag source IPs with RST count above 99th percentile
rst_counts    = df_clean[df_clean['flag_rst'] == 1].groupby('Source').size()
rst_threshold = rst_counts.quantile(0.99)

high_rst_sources = rst_counts[rst_counts > rst_threshold].index
df_clean['outlier_rst_flood'] = df_clean['Source'].isin(high_rst_sources).astype(int)

# Malformed packets 
df_clean['outlier_malformed'] = (
    df_clean['Info'].str.contains('Malformed', case=False, na=False)
).astype(int)

# Combined anomaly flag
outlier_cols = [
    'outlier_size', 'outlier_win', 'outlier_zero_win',
    'outlier_flags', 'outlier_syn_scan', 'outlier_dst_diversity',
    'outlier_rst_flood', 'outlier_malformed'
]
df_clean['is_anomaly'] = (df_clean[outlier_cols].sum(axis=1) > 0).astype(int)

# Outlier Results

print(" OUTLIER DETECTION RESULTS ")
print()

total = len(df_clean)
for col in outlier_cols:
    count = df_clean[col].sum()
    pct   = count / total * 100
    print(f"  {col:<28} {count:>8,}  ({pct:.2f}%)")

print()
print(f"  {'Total anomalous packets':<28} {df_clean['is_anomaly'].sum():>8,}  ({df_clean['is_anomaly'].mean()*100:.2f}%)")
print(f"  {'Total normal packets':<28} {(df_clean['is_anomaly']==0).sum():>8,}  ({(df_clean['is_anomaly']==0).mean()*100:.2f}%)")
print()

# breakdown of anomalies by protocol
print(" ANOMALIES BY PROTOCOL GROUP ")
print(df_clean.groupby('protocol_group')['is_anomaly'].agg(['sum', 'mean']).rename(
    columns={'sum': 'anomaly_count', 'mean': 'anomaly_rate'}
).round(4).to_string())
print()

# top sources flagged as anomalous
print(" TOP 10 ANOMALOUS SOURCE IPs ")
anomalous = df_clean[df_clean['is_anomaly'] == 1]
top_anomalous = anomalous.groupby('Source').agg(
    total_packets    = ('Length', 'count'),
    anomaly_triggers = (outlier_cols[0], lambda x: 
        df_clean.loc[x.index, outlier_cols].sum().to_dict()
    )
).sort_values('total_packets', ascending=False).head(10)
print(anomalous['Source'].value_counts().head(10).to_string())

# Outlier Visualisations

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Exploratory Anomaly Detection', fontsize=14)

# Outlier type counts
outlier_counts = df_clean[outlier_cols].sum().sort_values(ascending=False)
outlier_counts.plot(kind='barh', ax=axes[0,0], color='crimson', edgecolor='white')
axes[0,0].set_title('Outlier Count by Type')
axes[0,0].set_xlabel('Packet Count')
axes[0,0].invert_yaxis()

# Normal vs anomaly
pd.Series({
    'Normal':   (df_clean['is_anomaly'] == 0).sum(),
    'Anomaly':  (df_clean['is_anomaly'] == 1).sum()
}).plot(kind='bar', ax=axes[0,1], color=['steelblue', 'crimson'], edgecolor='white')
axes[0,1].set_title('Normal vs Anomalous Packets')
axes[0,1].set_ylabel('Packet Count')
axes[0,1].tick_params(axis='x', rotation=0)

# Anomaly rate by protocol
anomaly_by_proto = df_clean.groupby('protocol_group')['is_anomaly'].mean() * 100
anomaly_by_proto.plot(kind='bar', ax=axes[1,0], color='crimson', edgecolor='white')
axes[1,0].set_title('Anomaly Rate by Protocol (%)')
axes[1,0].set_xlabel('Protocol Group')
axes[1,0].set_ylabel('Anomaly Rate (%)')
axes[1,0].tick_params(axis='x', rotation=0)

# Top 10 anomalous source IPs
anomalous['Source'].value_counts().head(10).plot(
    kind='barh', ax=axes[1,1], color='crimson', edgecolor='white'
)
axes[1,1].set_title('Top 10 Anomalous Source IPs')
axes[1,1].set_xlabel('Anomalous Packet Count')
axes[1,1].invert_yaxis()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# Investigate outlier_dst_diversity threshold

print(" DESTINATION DIVERSITY DISTRIBUTION ")
print()
print(host_dest_counts.describe().round(2))
print()
print("Percentiles:")
for p in [90, 95, 99, 99.5, 99.9]:
    print(f"  {p}th percentile: {host_dest_counts.quantile(p/100):.0f} unique destinations")
print()
print(f"Current threshold (99th pct): {dst_threshold:.0f}")
print(f"Hosts above threshold:        {len(high_dst_sources):,}")
print()
print("Top 20 sources by unique destinations:")
print(host_dest_counts.sort_values(ascending=False).head(20).to_string())

# Investigate outlier_rst_flood threshold

print(" RST FLOOD DISTRIBUTION ")
print()
print(rst_counts.describe().round(2))
print()
print("Percentiles:")
for p in [90, 95, 99, 99.5, 99.9]:
    print(f"  {p}th percentile: {rst_counts.quantile(p/100):.0f} RST packets")
print()
print(f"Current threshold (99th pct): {rst_threshold:.0f}")
print(f"Sources above threshold:      {len(high_rst_sources):,}")
print()
print("Top 10 sources by RST count:")
print(rst_counts.sort_values(ascending=False).head(10).to_string())

# COMMAND ----------

#  Recalibrate Outlier Thresholds

# 
dst_threshold    = host_dest_counts.quantile(0.999)
high_dst_sources = host_dest_counts[host_dest_counts > dst_threshold].index
df_clean['outlier_dst_diversity'] = df_clean['Source'].isin(high_dst_sources).astype(int)

print(f"dst_diversity new threshold:  {dst_threshold:.0f} unique destinations")
print(f"Hosts above threshold:        {len(high_dst_sources):,}")
print(f"Packets flagged:              {df_clean['outlier_dst_diversity'].sum():,} ({df_clean['outlier_dst_diversity'].mean()*100:.2f}%)")
print()
print("Hosts flagged:")
print(host_dest_counts[host_dest_counts > dst_threshold].sort_values(ascending=False).to_string())
print()

# Recalibrate rst_flood 
rst_threshold    = rst_counts.quantile(0.999)
high_rst_sources = rst_counts[rst_counts > rst_threshold].index
df_clean['outlier_rst_flood'] = df_clean['Source'].isin(high_rst_sources).astype(int)

print(f"rst_flood new threshold:      {rst_threshold:.0f} RST packets")
print(f"Sources above threshold:      {len(high_rst_sources):,}")
print(f"Packets flagged:              {df_clean['outlier_rst_flood'].sum():,} ({df_clean['outlier_rst_flood'].mean()*100:.2f}%)")
print()
print("Sources flagged:")
print(rst_counts[rst_counts > rst_threshold].sort_values(ascending=False).to_string())
print()

# Recalibrate syn_scan 
syn_threshold    = syn_dst_counts.quantile(0.999)
high_syn_sources = syn_dst_counts[syn_dst_counts > syn_threshold].index
df_clean['outlier_syn_scan'] = df_clean['Source'].isin(high_syn_sources).astype(int)

print(f"syn_scan new threshold:       {syn_threshold:.0f} unique SYN destinations")
print(f"Sources above threshold:      {len(high_syn_sources):,}")
print(f"Packets flagged:              {df_clean['outlier_syn_scan'].sum():,} ({df_clean['outlier_syn_scan'].mean()*100:.2f}%)")
print()
print("Sources flagged:")
print(syn_dst_counts[syn_dst_counts > syn_threshold].sort_values(ascending=False).to_string())
print()

#  Recalculate combined anomaly flag 
outlier_cols = [
    'outlier_size', 'outlier_win', 'outlier_zero_win',
    'outlier_flags', 'outlier_syn_scan', 'outlier_dst_diversity',
    'outlier_rst_flood', 'outlier_malformed'
]
df_clean['is_anomaly'] = (df_clean[outlier_cols].sum(axis=1) > 0).astype(int)

print(" UPDATED OUTLIER DETECTION RESULTS ")
print()
total = len(df_clean)
for col in outlier_cols:
    count = df_clean[col].sum()
    pct   = count / total * 100
    print(f"  {col:<28} {count:>8,}  ({pct:.2f}%)")

print()
print(f"  {'Total anomalous packets':<28} {df_clean['is_anomaly'].sum():>8,}  ({df_clean['is_anomaly'].mean()*100:.2f}%)")
print(f"  {'Total normal packets':<28} {(df_clean['is_anomaly']==0).sum():>8,}  ({(df_clean['is_anomaly']==0).mean()*100:.2f}%)")

# COMMAND ----------

# Final Anomaly Visualisations

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Exploratory Anomaly Detection — Final Results', fontsize=14)

# Outlier type counts
outlier_counts = df_clean[outlier_cols].sum().sort_values(ascending=False)
outlier_counts.plot(kind='barh', ax=axes[0,0], color='crimson', edgecolor='white')
axes[0,0].set_title('Outlier Count by Type')
axes[0,0].set_xlabel('Packet Count')
axes[0,0].invert_yaxis()

# Normal vs anomaly
pd.Series({
    'Normal':  (df_clean['is_anomaly'] == 0).sum(),
    'Anomaly': (df_clean['is_anomaly'] == 1).sum()
}).plot(kind='bar', ax=axes[0,1], color=['steelblue', 'crimson'], edgecolor='white')
axes[0,1].set_title('Normal vs Anomalous Packets')
axes[0,1].set_ylabel('Packet Count')
axes[0,1].tick_params(axis='x', rotation=0)
for i, v in enumerate([(df_clean['is_anomaly']==0).sum(), (df_clean['is_anomaly']==1).sum()]):
    axes[0,1].text(i, v + 1000, f'{v:,}', ha='center', fontsize=9)

# Anomaly rate by protocol
anomaly_by_proto = df_clean.groupby('protocol_group')['is_anomaly'].mean() * 100
anomaly_by_proto.plot(kind='bar', ax=axes[1,0], color='crimson', edgecolor='white')
axes[1,0].set_title('Anomaly Rate by Protocol (%)')
axes[1,0].set_xlabel('Protocol Group')
axes[1,0].set_ylabel('Anomaly Rate (%)')
axes[1,0].tick_params(axis='x', rotation=0)
for i, v in enumerate(anomaly_by_proto):
    axes[1,0].text(i, v + 0.1, f'{v:.1f}%', ha='center', fontsize=9)

# Top 10 anomalous source IPs
anomalous = df_clean[df_clean['is_anomaly'] == 1]
anomalous['Source'].value_counts().head(10).plot(
    kind='barh', ax=axes[1,1], color='crimson', edgecolor='white'
)
axes[1,1].set_title('Top 10 Anomalous Source IPs')
axes[1,1].set_xlabel('Anomalous Packet Count')
axes[1,1].invert_yaxis()

plt.tight_layout()
plt.show()

# Save Labelled Dataset

df_clean.to_csv('pcap_0-1m_labelled.csv', index=False)
print(f"Saved: pcap_0-1m_labelled.csv")
print(f"Shape: {df_clean.shape}")
print()
print("Columns added during EDA:")
new_cols = ['size_category', 'protocol_group'] + outlier_cols + ['is_anomaly']
for col in new_cols:
    print(f"  {col}")

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Engineering

# COMMAND ----------

# Feature Engineering

# fill port NaN with -1 sentinel for ICMP
# -1 is outside valid port range 0-65535 so won't clash with real ports
df_clean['src_port_filled'] = df_clean['src_port'].fillna(-1).astype(int)
df_clean['dst_port_filled'] = df_clean['dst_port'].fillna(-1).astype(int)

# Flow key — using filled ports so ICMP flows are not dropped
flow_cols = ['Source', 'Destination', 'Protocol', 'src_port_filled', 'dst_port_filled']

# Inter-arrival time — must be computed before groupby
df_clean = df_clean.sort_values(flow_cols + ['Time']).reset_index(drop=True)
df_clean['iat'] = df_clean.groupby(flow_cols)['Time'].diff()

# Flow-based features
flow_features = df_clean.groupby(flow_cols).agg(
    flow_start     = ('Time',    'min'),
    flow_end       = ('Time',    'max'),
    packet_count   = ('Length',  'count'),
    total_bytes    = ('Length',  'sum'),
    avg_pkt_size   = ('Length',  'mean'),
    pkt_size_var   = ('Length',  'var'),
    mean_iat       = ('iat',     'mean'),
    std_iat        = ('iat',     'std'),
    syn_count      = ('flag_syn', 'sum'),
    ack_count      = ('flag_ack', 'sum'),
    fin_count      = ('flag_fin', 'sum'),
    rst_count      = ('flag_rst', 'sum'),
    well_known_dst = ('dst_port', lambda x: int(x.iloc[0] < 1024) if x.notna().any() else 0),
    well_known_src = ('src_port', lambda x: int(x.iloc[0] < 1024) if x.notna().any() else 0),
    protocol_group = ('protocol_group', 'first'),
    is_anomaly     = ('is_anomaly', 'max'),
).reset_index()

# Derived flow features
flow_features['flow_duration'] = flow_features['flow_end'] - flow_features['flow_start']

flow_features['pkts_per_sec']  = (
    flow_features['packet_count'] /
    flow_features['flow_duration'].replace(0, np.nan)
)
flow_features['bytes_per_sec'] = (
    flow_features['total_bytes'] /
    flow_features['flow_duration'].replace(0, np.nan)
)

# Bidirectional features
df_clean['canon_src']  = df_clean.apply(
    lambda r: r['Source'] if r['Source'] < r['Destination'] else r['Destination'], axis=1
)
df_clean['canon_dst']  = df_clean.apply(
    lambda r: r['Destination'] if r['Source'] < r['Destination'] else r['Source'], axis=1
)
df_clean['is_forward'] = (df_clean['Source'] == df_clean['canon_src']).astype(int)

fwd = df_clean[df_clean['is_forward'] == 1].groupby(flow_cols).agg(
    fwd_pkt_count = ('Length', 'count'),
    fwd_bytes     = ('Length', 'sum'),
).reset_index()

bwd = df_clean[df_clean['is_forward'] == 0].groupby(flow_cols).agg(
    bwd_pkt_count = ('Length', 'count'),
    bwd_bytes     = ('Length', 'sum'),
).reset_index()

flow_features = flow_features.merge(fwd, on=flow_cols, how='left')
flow_features = flow_features.merge(bwd, on=flow_cols, how='left')

flow_features[['fwd_pkt_count', 'fwd_bytes',
               'bwd_pkt_count', 'bwd_bytes']] = \
    flow_features[['fwd_pkt_count', 'fwd_bytes',
                   'bwd_pkt_count', 'bwd_bytes']].fillna(0)

flow_features['fwd_bwd_pkt_ratio']  = (
    flow_features['fwd_pkt_count'] /
    flow_features['bwd_pkt_count'].replace(0, np.nan)
)
flow_features['fwd_bwd_byte_ratio'] = (
    flow_features['fwd_bytes'] /
    flow_features['bwd_bytes'].replace(0, np.nan)
)

# Protocol encoding
protocol_encoding             = {proto: idx for idx, proto in enumerate(df_clean['Protocol'].unique())}
flow_features['protocol_encoded'] = flow_features['Protocol'].map(protocol_encoding)

proto_dummies = pd.get_dummies(flow_features['protocol_group'], prefix='proto').astype(int)
flow_features = pd.concat([flow_features, proto_dummies], axis=1)

# IP-level features
ip_features = df_clean.groupby('Source').agg(
    conn_diversity = ('Destination', 'nunique'),
    activity_level = ('Length',      'count'),
    unique_dst     = ('Destination', 'nunique'),
).reset_index()

flow_features = flow_features.merge(ip_features, on='Source', how='left')

flow_features['unique_src_ips'] = df_clean['Source'].nunique()
flow_features['unique_dst_ips'] = df_clean['Destination'].nunique()

# Summary
print(f"Total flows:    {len(flow_features):,}")
print(f"Total features: {len(flow_features.columns):,}")
print()
print("Protocol groups:")
print(flow_features['protocol_group'].value_counts())
print()
print("Proto dummy cols:", [c for c in flow_features.columns if c.startswith('proto_')])
print()
print("Columns:")
print(flow_features.columns.tolist())
print()
print(flow_features.head())

# COMMAND ----------

# Feature Engineering - NaN Check

print(" NaN COUNTS ")
nan_counts = flow_features.isnull().sum()
nan_pct    = (nan_counts / len(flow_features) * 100).round(2)

nan_summary = pd.DataFrame({
    'nan_count': nan_counts,
    'nan_pct':   nan_pct,
    'dtype':     flow_features.dtypes
})

nan_summary = nan_summary[nan_summary['nan_count'] > 0].sort_values('nan_count', ascending=False)

if len(nan_summary) == 0:
    print("No NaN values found")
else:
    print(nan_summary.to_string())

# COMMAND ----------

# Feature Engineering - Fill NaNs

# fwd_bwd ratios — null means no backward traffic
# fill with forward count/bytes meaning 100% unidirectional
flow_features['fwd_bwd_pkt_ratio']  = flow_features['fwd_bwd_pkt_ratio'].fillna(
    flow_features['fwd_pkt_count']
)
flow_features['fwd_bwd_byte_ratio'] = flow_features['fwd_bwd_byte_ratio'].fillna(
    flow_features['fwd_bytes']
)

# single packet flows — no inter-arrival time or variance possible
# fill with 0 — meaning no variation observed
flow_features['std_iat']     = flow_features['std_iat'].fillna(0)
flow_features['mean_iat']    = flow_features['mean_iat'].fillna(0)
flow_features['pkt_size_var']= flow_features['pkt_size_var'].fillna(0)

# zero duration flows — pkts/bytes per sec undefined / fill with packet_count/total_bytes — treats the flow as 1 second long
flow_features['pkts_per_sec']  = flow_features['pkts_per_sec'].fillna(
    flow_features['packet_count']
)
flow_features['bytes_per_sec'] = flow_features['bytes_per_sec'].fillna(
    flow_features['total_bytes']
)

# verify
remaining = flow_features.isnull().sum()
remaining = remaining[remaining > 0]
if len(remaining) == 0:
    print("No NaN values remaining")
else:
    print("Remaining NaNs:")
    print(remaining.to_string())

print()
print(f"Flow features shape: {flow_features.shape}")

# COMMAND ----------

# Feature Engineering - Standardisation for SVM

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pickle

#  Select numerical features for SVM 
exclude_cols = [
    'Source', 'Destination', 'Protocol', 'protocol_group',
    'src_port_filled', 'dst_port_filled',
    'flow_start', 'flow_end',
    'is_anomaly',
    'well_known_dst', 'well_known_src',
    'proto_ICMP', 'proto_TCP', 'proto_UDP',
    'protocol_encoded',
]

feature_cols = [
    c for c in flow_features.columns
    if c not in exclude_cols
    and pd.api.types.is_numeric_dtype(flow_features[c])
]

# Drop zero/near-zero variance features 
# first pass — strict zero variance
selector     = VarianceThreshold(threshold=0)
selector.fit(flow_features[feature_cols])
feature_cols = [c for c, keep in zip(feature_cols, selector.get_support()) if keep]

# second pass — near-zero variance (std < 0.01 after scaling)
# fit a temporary scaler to detect these
_X_temp   = StandardScaler().fit_transform(flow_features[feature_cols])
low_std   = [col for col, std in zip(feature_cols, _X_temp.std(axis=0)) if std < 0.01]

if low_std:
    print(f"Dropping near-zero variance features: {low_std}")
    feature_cols = [c for c in feature_cols if c not in low_std]
else:
    print("No near-zero variance features found")

print()
print(" FEATURES SELECTED FOR SVM ")
print(f"Total features: {len(feature_cols)}")
print()
for col in feature_cols:
    print(f"  {col}")
print()

# Standardise 
X        = flow_features[feature_cols]
y        = flow_features['is_anomaly']
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(" STANDARDISATION COMPLETE ")
print(f"X shape:  {X_scaled.shape}")
print(f"y shape:  {y.shape}")
print()
print("Class distribution:")
print(f"  Normal (0):  {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%)")
print(f"  Anomaly (1): {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)")
print()
print("Feature means after scaling (should be ~0):")
print(pd.Series(X_scaled.mean(axis=0).round(4), index=feature_cols).to_string())
print()
print("Feature stds after scaling (should be ~1):")
print(pd.Series(X_scaled.std(axis=0).round(4), index=feature_cols).to_string())

# Save 
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

X_df               = pd.DataFrame(X_scaled, columns=feature_cols)
X_df['is_anomaly'] = y.values
X_df.to_csv('flow_features_scaled.csv', index=False)

print()
print("Saved: scaler.pkl")
print("Saved: flow_features_scaled.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# SVD - Dimensionality Reduction

from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np

# Fit SVD with all 22 components first to see explained variance 
svd_full     = TruncatedSVD(n_components=21, random_state=42)  # max is n_features - 1
svd_full.fit(X_scaled)

explained_variance      = svd_full.explained_variance_ratio_
cumulative_variance     = np.cumsum(explained_variance)

print(" SVD EXPLAINED VARIANCE ")
print()
print(f"{'Component':<12} {'Explained %':<15} {'Cumulative %'}")
print("-" * 40)
for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
    marker = " ◄" if cv >= 0.95 and (i == 0 or cumulative_variance[i-1] < 0.95) else ""
    print(f"  {i+1:<10} {ev*100:<15.2f} {cv*100:.2f}%{marker}")

# Plot scree 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('SVD Explained Variance', fontsize=13)

axes[0].bar(range(1, 22), explained_variance * 100, color='steelblue', edgecolor='white')
axes[0].set_title('Explained Variance per Component')
axes[0].set_xlabel('Component')
axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_xticks(range(1, 22))

axes[1].plot(range(1, 22), cumulative_variance * 100, marker='o', color='steelblue')
axes[1].axhline(y=95, color='crimson', linestyle='--', label='95% threshold')
axes[1].axhline(y=99, color='orange',  linestyle='--', label='99% threshold')
axes[1].set_title('Cumulative Explained Variance')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].set_xticks(range(1, 22))
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 13 is the standard choice because it's the point where cumulative variance crosses 95% — the conventional threshold for dimensionality reduction here 13 components capture 95.99% of variance. Components 18-21 are zero meaning there are only 17 meaningful dimensions in the data despite having 22 features, indicateing some features are correlated.

# COMMAND ----------

# SVD - Apply Final Transformation

n_components = 13  # captures 95.99% of variance

svd = TruncatedSVD(n_components=n_components, random_state=42)
X_svd = svd.fit_transform(X_scaled)

print(" SVD TRANSFORMATION COMPLETE ")
print(f"Original shape:  {X_scaled.shape}")
print(f"Reduced shape:   {X_svd.shape}")
print(f"Variance captured: {svd.explained_variance_ratio_.sum()*100:.2f}%")
print()

# Component composition / shows which original features drive each component
components_df = pd.DataFrame(
    svd.components_,
    columns=feature_cols,
    index=[f'SVD_{i+1}' for i in range(n_components)]
)

print(" TOP 3 FEATURES PER COMPONENT ")
for idx, row in components_df.iterrows():
    top3 = row.abs().nlargest(3).index.tolist()
    print(f"  {idx}: {top3}")
print()

# Save 
import pickle

with open('svd.pkl', 'wb') as f:
    pickle.dump(svd, f)

X_svd_df               = pd.DataFrame(X_svd, columns=[f'SVD_{i+1}' for i in range(n_components)])
X_svd_df['is_anomaly'] = y.values
X_svd_df.to_csv('flow_features_svd.csv', index=False)

print("Saved: svd.pkl")
print("Saved: flow_features_svd.csv")
print()
print(f"Ready for SVM — X shape: {X_svd.shape}, y shape: {y.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC Maschine Learning Models - SVM clsaaification with 80/20 ratio

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# Model Training - Train/Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_svd, y.values,
    test_size=0.2,
    random_state=42,
    stratify=y.values  # preserve 80/20 class ratio in both splits
)

print(" TRAIN/TEST SPLIT COMPLETE")
print(f"Train set: {X_train.shape[0]:,} flows ({X_train.shape[0]/len(X_svd)*100:.0f}%)")
print(f"Test set:  {X_test.shape[0]:,} flows ({X_test.shape[0]/len(X_svd)*100:.0f}%)")
print()
print("Class distribution — Train:")
print(f"  Normal (0):  {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.2f}%)")
print(f"  Anomaly (1): {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.2f}%)")
print()
print("Class distribution — Test:")
print(f"  Normal (0):  {(y_test == 0).sum():,} ({(y_test == 0).mean()*100:.2f}%)")
print(f"  Anomaly (1): {(y_test == 1).sum():,} ({(y_test == 1).mean()*100:.2f}%)")

#  Model Training - SVM

from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay)
import time

print(" TRAINING SVM (RBF) ")
print("This may take a few minutes on large amount of flows...")
print()

svm_start = time.time()

svm = SVC(
    kernel='rbf',
    class_weight='balanced',
    probability=True,   # needed for ROC AUC
    random_state=42
)
svm.fit(X_train, y_train)

svm_time = time.time() - svm_start
print(f"Training time: {svm_time:.2f} seconds")
print()

# Evaluate Model
y_pred_svm  = svm.predict(X_test)
y_prob_svm  = svm.predict_proba(X_test)[:, 1]
auc_svm     = roc_auc_score(y_test, y_prob_svm)

print(" SVM RESULTS ")
print()
print(classification_report(y_test, y_pred_svm, target_names=['Normal', 'Anomaly']))

# COMMAND ----------

# MAGIC %md
# MAGIC SVM Performance
# MAGIC MetricNormalAnomalyPrecision0.990.92Recall0.980.96F1 Score0.980.94Accuracy0.97
# MAGIC These are very strong results. Key points:
# MAGIC
# MAGIC Recall 0.96 on Anomaly — the model correctly identifies 96% of all anomalous flows, missing only 4%. For intrusion detection this is the most important metric
# MAGIC Precision 0.92 on Anomaly — 8% false positive rate, meaning some normal flows are flagged as anomalous — acceptable for this use case
# MAGIC 400 seconds training time 

# COMMAND ----------

# Model Training - Naive Bayes

from sklearn.naive_bayes import GaussianNB

print(" TRAINING NAIVE BAYES ")
print()

nb_start = time.time()

nb = GaussianNB()
nb.fit(X_train, y_train)

nb_time = time.time() - nb_start
print(f"Training time: {nb_time:.2f} seconds")
print()

# Evaluate 
y_pred_nb = nb.predict(X_test)
y_prob_nb = nb.predict_proba(X_test)[:, 1]
auc_nb    = roc_auc_score(y_test, y_prob_nb)

print(" NAIVE BAYES RESULTS ")
print()
print(classification_report(y_test, y_pred_nb, target_names=['Normal', 'Anomaly']))
print(f"ROC AUC: {auc_nb:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC Compare SVM vs NB
# MAGIC | Metric                | SVM   | Naive Bayes |
# MAGIC |-----------------------|-------|-------------|
# MAGIC | Accuracy              | 0.97  | 0.88        |
# MAGIC | Precision (Anomaly)   | 0.92  | 0.76        |
# MAGIC | Recall (Anomaly)      | 0.96  | 0.56        |
# MAGIC | F1 (Anomaly)          | 0.94  | 0.65        |
# MAGIC | ROC AUC               | —     | 0.93        |

# COMMAND ----------

# Model Comparison - Visualisations

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('SVM vs Naive Bayes — Model Comparison', fontsize=14)

# Confusion matrices 
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_svm),
    display_labels=['Normal', 'Anomaly']
).plot(ax=axes[0, 0], colorbar=False, cmap='Blues')
axes[0, 0].set_title('SVM — Confusion Matrix')

ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_nb),
    display_labels=['Normal', 'Anomaly']
).plot(ax=axes[0, 1], colorbar=False, cmap='Blues')
axes[0, 1].set_title('Naive Bayes — Confusion Matrix')

# ROC curves 
from sklearn.metrics import roc_curve

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
fpr_nb,  tpr_nb,  _ = roc_curve(y_test, y_prob_nb)

axes[0, 2].plot(fpr_svm, tpr_svm, label=f'SVM  (AUC={auc_svm:.3f})', color='steelblue')
axes[0, 2].plot(fpr_nb,  tpr_nb,  label=f'NB   (AUC={auc_nb:.3f})',  color='crimson')
axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[0, 2].set_title('ROC Curve')
axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Metric comparison bar chart 
from sklearn.metrics import precision_score, recall_score, f1_score

metrics = {
    'Precision': [
        precision_score(y_test, y_pred_svm),
        precision_score(y_test, y_pred_nb)
    ],
    'Recall': [
        recall_score(y_test, y_pred_svm),
        recall_score(y_test, y_pred_nb)
    ],
    'F1 Score': [
        f1_score(y_test, y_pred_svm),
        f1_score(y_test, y_pred_nb)
    ],
    'ROC AUC': [auc_svm, auc_nb]
}

x     = np.arange(len(metrics))
width = 0.35

bars1 = axes[1, 0].bar(x - width/2, [v[0] for v in metrics.values()],
                        width, label='SVM', color='steelblue', edgecolor='white')
bars2 = axes[1, 0].bar(x + width/2, [v[1] for v in metrics.values()],
                        width, label='NB',  color='crimson',   edgecolor='white')

axes[1, 0].set_title('Metric Comparison')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics.keys())
axes[1, 0].set_ylim(0, 1.1)
axes[1, 0].legend()
axes[1, 0].set_ylabel('Score')
for bar in bars1:
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}', ha='center', fontsize=8)
for bar in bars2:
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}', ha='center', fontsize=8)

# Training time comparison 
axes[1, 1].bar(['SVM', 'Naive Bayes'], [svm_time, nb_time],
               color=['steelblue', 'crimson'], edgecolor='white')
axes[1, 1].set_title('Training Time (seconds)')
axes[1, 1].set_ylabel('Seconds')
for i, v in enumerate([svm_time, nb_time]):
    axes[1, 1].text(i, v + 0.1, f'{v:.2f}s', ha='center', fontsize=9)

#  Summary table 
axes[1, 2].axis('off')
summary_data = [
    ['Metric',      'SVM',                                    'Naive Bayes'],
    ['Precision',   f"{precision_score(y_test, y_pred_svm):.4f}",
                    f"{precision_score(y_test, y_pred_nb):.4f}"],
    ['Recall',      f"{recall_score(y_test, y_pred_svm):.4f}",
                    f"{recall_score(y_test, y_pred_nb):.4f}"],
    ['F1 Score',    f"{f1_score(y_test, y_pred_svm):.4f}",
                    f"{f1_score(y_test, y_pred_nb):.4f}"],
    ['ROC AUC',     f"{auc_svm:.4f}",                         f"{auc_nb:.4f}"],
    ['Train Time',  f"{svm_time:.2f}s",                       f"{nb_time:.2f}s"],
]
table = axes[1, 2].table(cellText=summary_data[1:], colLabels=summary_data[0],
                          loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
axes[1, 2].set_title('Summary Table')

plt.tight_layout()
plt.show()

# Save models 
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

with open('nb_model.pkl', 'wb') as f:
    pickle.dump(nb, f)

print("Saved: svm_model.pkl")
print("Saved: nb_model.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC Confusion matrix breakdown
# MAGIC SVM:
# MAGIC
# MAGIC 15,321 true negatives — normal flows correctly identified
# MAGIC 3,661 true positives — anomalies correctly caught
# MAGIC 152 false negatives — anomalies missed
# MAGIC 340 false positives — normal flows incorrectly flagged
# MAGIC
# MAGIC NB:
# MAGIC
# MAGIC 14,983 true negatives
# MAGIC 2,144 true positives
# MAGIC 1,669 false negatives — missed 44% of anomalies
# MAGIC 678 false positives
# MAGIC
# MAGIC ROC curve — SVM (0.985) outperforms NB (0.932) across all thresholds. Both are well above the diagonal baseline.
# MAGIC What to write in your report:
# MAGIC
# MAGIC SVM with RBF kernel significantly outperformed Gaussian Naive Bayes across all metrics. SVM achieved a recall of 0.960 on anomalous flows, correctly identifying 3,661 of 3,813 anomalies in the test set, compared to Naive Bayes which detected only 2,144 (56.2%). The ROC AUC of 0.985 for SVM indicates near-perfect discriminative ability. The primary tradeoff is computational cost — SVM required 400 seconds to train versus 0.02 seconds for Naive Bayes, a 20,000x difference. For a production intrusion detection system where detection rate is critical, SVM is the clear choice. Naive Bayes remains viable for resource-constrained environments where training speed is prioritised over recall.

# COMMAND ----------

# MAGIC %md
# MAGIC Cross validation - 5-fold CV means training SVM 5 times

# COMMAND ----------

# Cross-Validation - SVM and Naive Bayes

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import numpy as np

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# SVM cross-validation 
print(" SVM 5-FOLD CROSS-VALIDATION ")
print("This will take several minutes...")
print()

svm_cv = cross_validate(
    SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    X_svd, y.values,
    cv=cv,
    scoring=scoring,
    n_jobs=-1  # use all cores
)

for metric in scoring:
    scores = svm_cv[f'test_{metric}']
    print(f"  {metric:<12} {scores.mean():.4f} (+/- {scores.std():.4f})  {np.round(scores, 4)}")

print()

#  NB cross-validation 
print(" NAIVE BAYES 5-FOLD CROSS-VALIDATION ")
print()

nb_cv = cross_validate(
    GaussianNB(),
    X_svd, y.values,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

for metric in scoring:
    scores = nb_cv[f'test_{metric}']
    print(f"  {metric:<12} {scores.mean():.4f} (+/- {scores.std():.4f})  {np.round(scores, 4)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### SVM — CV vs Holdout comparison
# MAGIC
# MAGIC | Metric     | Holdout | CV Mean | CV Std    | Verdict     |
# MAGIC |------------|---------|---------|-----------|-------------|
# MAGIC | Accuracy   | 0.9700  | 0.9763  | ±0.0018   | Consistent|
# MAGIC | Precision  | 0.9150  | 0.9245  | ±0.0034   | Consistent|
# MAGIC | Recall     | 0.9601  | 0.9571  | ±0.0062   | Consistent|
# MAGIC | F1         | 0.9370  | 0.9405  | ±0.0046   | Consistent|
# MAGIC | ROC AUC    | 0.9850  | 0.9854  | ±0.0021   | Consistent|
# MAGIC
# MAGIC Very low standard deviations across all folds — the model is stable not overfitted.
# MAGIC
# MAGIC ### Naive Bayes — CV vs Holdout comparison
# MAGIC
# MAGIC | Metric     | Holdout | CV Mean | CV Std    | Verdict     |
# MAGIC |------------|---------|---------|-----------|-------------|
# MAGIC | Accuracy   | 0.8800  | 0.8799  | ±0.0161   | Consistent|
# MAGIC | Precision  | 0.7597  | 0.7543  | ±0.0187   | Consistent|
# MAGIC | Recall     | 0.5623  | 0.5695  | ±0.0924   | High variance|
# MAGIC | F1         | 0.6463  | 0.6461  | ±0.0644   | High variance|
# MAGIC | ROC AUC    | 0.9320  | 0.9290  | ±0.0043   | Consistent|
# MAGIC
# MAGIC NB recall std of ±0.0924 is notable — fold 2 scored 0.74 while fold 5 scored only 0.46. This means NB performance is sensitive to which data it trains on, confirming it's less robust than SVM for this task.
# MAGIC
# MAGIC
# MAGIC 5-fold stratified cross-validation confirmed that both models generalise well to unseen data. SVM achieved a mean CV F1 of 0.9405 (±0.0046) and ROC AUC of 0.9854 (±0.0021), with low variance across all folds indicating a stable, well-generalised model with no evidence of overfitting. Naive Bayes showed higher variance in recall (±0.0924), suggesting its performance is more sensitive to the training data distribution, likely due to the feature independence assumption being violated in this dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi file Pipline (which can be run separetly)
# MAGIC
# MAGIC As the spark clustering is not supported for free acount below is an attempt to run larger dataset - approx 2 milion rows (reduced from 4 as i was having problems with available memory)
# MAGIC
# MAGIC -  Multi-file parsing  
# MAGIC -  Outlier detection
# MAGIC -  Feature engineering  
# MAGIC -  Standardisation  
# MAGIC -  SVD  
# MAGIC -  SVM + Naive Bayes  
# MAGIC -  Compare single file vs multi-file results
# MAGIC

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, roc_auc_score, precision_score, recall_score, f1_score)
import pickle
import time
import gc


# Parse All Files

import pandas as pd
import numpy as np

files = [
    'pcap_0-1m.csv',
    'pcap_2m-4m.csv',
 
]

parsed_frames = []

for f in files:
    print(f"Parsing {f}...")
    df = pd.read_csv(f, encoding='latin-1')
    df = expand_info_column(df)
    df = df[~df['protocol_group'].isin(['DROP', 'OTHER'])].reset_index(drop=True)
    df = df.drop_duplicates(
        subset=['Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info']
    ).reset_index(drop=True)
    flag_cols_all = ['flag_syn', 'flag_ack', 'flag_psh', 'flag_fin',
                     'flag_rst', 'flag_urg', 'flag_ecn', 'flag_cwr', 'flag_ae']
    df[flag_cols_all] = df[flag_cols_all].fillna(0)
    df['src_port_filled'] = df['src_port'].fillna(-1).astype(int)
    df['dst_port_filled'] = df['dst_port'].fillna(-1).astype(int)
    parsed_frames.append(df)
    print(f"  {len(df):,} rows after cleaning")

print()
print("Concatenating...")
df_all = pd.concat(parsed_frames, ignore_index=True)
print(f"Total rows: {len(df_all):,}")
print()
print("Protocol groups:")
print(df_all['protocol_group'].value_counts())

# Outlier Detection

tcp_mask = df_all['protocol_group'] == 'TCP'

df_all['outlier_size']     = ((df_all['Length'] < 32) | (df_all['Length'] > 1504)).astype(int)
df_all['outlier_zero_win'] = 0
df_all.loc[tcp_mask, 'outlier_zero_win'] = (df_all.loc[tcp_mask, 'win'] == 0).astype(int)
df_all['outlier_flags']    = 0
df_all.loc[tcp_mask, 'outlier_flags'] = (
    ((df_all.loc[tcp_mask, 'flag_syn'] == 1) & (df_all.loc[tcp_mask, 'flag_fin'] == 1)) |
    (df_all.loc[tcp_mask, 'flag_urg'] == 1) |
    (df_all.loc[tcp_mask, 'flag_ae']  == 1)
).astype(int)

# syn scan
syn_packets      = df_all[(df_all['protocol_group'] == 'TCP') & (df_all['flag_syn'] == 1)]
syn_dst_counts   = syn_packets.groupby('Source')['Destination'].nunique()
syn_threshold    = syn_dst_counts.quantile(0.999)
high_syn         = syn_dst_counts[syn_dst_counts > syn_threshold].index
df_all['outlier_syn_scan'] = df_all['Source'].isin(high_syn).astype(int)

# dst diversity
host_dest_counts = df_all.groupby('Source')['Destination'].nunique()
dst_threshold    = host_dest_counts.quantile(0.999)
high_dst         = host_dest_counts[host_dest_counts > dst_threshold].index
df_all['outlier_dst_diversity'] = df_all['Source'].isin(high_dst).astype(int)

# rst flood
rst_counts       = df_all[df_all['flag_rst'] == 1].groupby('Source').size()
rst_threshold    = rst_counts.quantile(0.999)
high_rst         = rst_counts[rst_counts > rst_threshold].index
df_all['outlier_rst_flood'] = df_all['Source'].isin(high_rst).astype(int)

outlier_cols_all = ['outlier_size', 'outlier_zero_win', 'outlier_flags',
                    'outlier_syn_scan', 'outlier_dst_diversity', 'outlier_rst_flood']

df_all['is_anomaly'] = (df_all[outlier_cols_all].sum(axis=1) > 0).astype(int)

print(" OUTLIER DETECTION RESULTS (2 FILES) ")
total = len(df_all)
for col in outlier_cols_all:
    count = df_all[col].sum()
    print(f"  {col:<28} {count:>10,}  ({count/total*100:.2f}%)")
print()
print(f"  Total anomalous: {df_all['is_anomaly'].sum():,} ({df_all['is_anomaly'].mean()*100:.2f}%)")
print(f"  Total normal:    {(df_all['is_anomaly']==0).sum():,} ({(df_all['is_anomaly']==0).mean()*100:.2f}%)")

# drop columns not needed for flow features
drop_cols = [c for c in df_all.columns if c not in [
    'Time', 'Source', 'Destination', 'Protocol', 'Length',
    'protocol_group', 'src_port_filled', 'dst_port_filled',
    'flag_syn', 'flag_ack', 'flag_psh', 'flag_fin',
    'flag_rst', 'flag_urg', 'flag_ecn', 'flag_cwr', 'flag_ae',
    'win', 'src_port', 'dst_port', 'is_anomaly',
    'outlier_size', 'outlier_zero_win', 'outlier_flags',
    'outlier_syn_scan', 'outlier_dst_diversity', 'outlier_rst_flood',
]]
df_all = df_all.drop(columns=drop_cols)
gc.collect()

print(f"Memory freed. df_all shape: {df_all.shape}")


keep = [
    'Time', 'Source', 'Destination', 'Protocol', 'Length',
    'protocol_group', 'src_port_filled', 'dst_port_filled',
    'flag_syn', 'flag_ack', 'flag_psh', 'flag_fin',
    'flag_rst', 'flag_urg', 'flag_ecn', 'flag_cwr', 'flag_ae',
    'win', 'src_port', 'dst_port', 'is_anomaly',
    'outlier_size', 'outlier_zero_win', 'outlier_flags',
    'outlier_syn_scan', 'outlier_dst_diversity', 'outlier_rst_flood',
]

drop_cols = [c for c in df_all.columns if c not in keep]
df_all    = df_all.drop(columns=drop_cols)
gc.collect()

print(f"Memory freed. df_all shape: {df_all.shape}")
print(f"Columns remaining: {len(df_all.columns)}")

# Feature Engineering

flow_cols = ['Source', 'Destination', 'Protocol', 'src_port_filled', 'dst_port_filled']

df_all = df_all.sort_values(flow_cols + ['Time']).reset_index(drop=True)
df_all['iat'] = df_all.groupby(flow_cols)['Time'].diff()

flow_features_all = df_all.groupby(flow_cols).agg(
    flow_start     = ('Time',     'min'),
    flow_end       = ('Time',     'max'),
    packet_count   = ('Length',   'count'),
    total_bytes    = ('Length',   'sum'),
    avg_pkt_size   = ('Length',   'mean'),
    pkt_size_var   = ('Length',   'var'),
    mean_iat       = ('iat',      'mean'),
    std_iat        = ('iat',      'std'),
    syn_count      = ('flag_syn', 'sum'),
    ack_count      = ('flag_ack', 'sum'),
    fin_count      = ('flag_fin', 'sum'),
    rst_count      = ('flag_rst', 'sum'),
    well_known_dst = ('dst_port', lambda x: int(x.iloc[0] < 1024) if x.notna().any() else 0),
    well_known_src = ('src_port', lambda x: int(x.iloc[0] < 1024) if x.notna().any() else 0),
    protocol_group = ('protocol_group', 'first'),
    is_anomaly     = ('is_anomaly', 'max'),
).reset_index()

flow_features_all['flow_duration'] = flow_features_all['flow_end'] - flow_features_all['flow_start']
flow_features_all['pkts_per_sec']  = (
    flow_features_all['packet_count'] /
    flow_features_all['flow_duration'].replace(0, np.nan)
)
flow_features_all['bytes_per_sec'] = (
    flow_features_all['total_bytes'] /
    flow_features_all['flow_duration'].replace(0, np.nan)
)

# bidirectional
df_all['canon_src']  = df_all.apply(
    lambda r: r['Source'] if r['Source'] < r['Destination'] else r['Destination'], axis=1)
df_all['is_forward'] = (df_all['Source'] == df_all['canon_src']).astype(int)

fwd = df_all[df_all['is_forward'] == 1].groupby(flow_cols).agg(
    fwd_pkt_count=('Length', 'count'),
    fwd_bytes    =('Length', 'sum')).reset_index()
bwd = df_all[df_all['is_forward'] == 0].groupby(flow_cols).agg(
    bwd_pkt_count=('Length', 'count'),
    bwd_bytes    =('Length', 'sum')).reset_index()

flow_features_all = flow_features_all.merge(fwd, on=flow_cols, how='left')
flow_features_all = flow_features_all.merge(bwd, on=flow_cols, how='left')
flow_features_all[['fwd_pkt_count', 'fwd_bytes',
                    'bwd_pkt_count', 'bwd_bytes']] = \
    flow_features_all[['fwd_pkt_count', 'fwd_bytes',
                        'bwd_pkt_count', 'bwd_bytes']].fillna(0)

flow_features_all['fwd_bwd_pkt_ratio']  = (
    flow_features_all['fwd_pkt_count'] /
    flow_features_all['bwd_pkt_count'].replace(0, np.nan)
)
flow_features_all['fwd_bwd_byte_ratio'] = (
    flow_features_all['fwd_bytes'] /
    flow_features_all['bwd_bytes'].replace(0, np.nan)
)

# IP level
ip_features = df_all.groupby('Source').agg(
    conn_diversity=('Destination', 'nunique'),
    activity_level=('Length',      'count'),
    unique_dst    =('Destination', 'nunique'),
).reset_index()
flow_features_all = flow_features_all.merge(ip_features, on='Source', how='left')

# protocol encoding
proto_dummies     = pd.get_dummies(flow_features_all['protocol_group'], prefix='proto').astype(int)
flow_features_all = pd.concat([flow_features_all, proto_dummies], axis=1)

# fill NaNs
flow_features_all['fwd_bwd_pkt_ratio']  = flow_features_all['fwd_bwd_pkt_ratio'].fillna(
    flow_features_all['fwd_pkt_count'])
flow_features_all['fwd_bwd_byte_ratio'] = flow_features_all['fwd_bwd_byte_ratio'].fillna(
    flow_features_all['fwd_bytes'])
flow_features_all['std_iat']       = flow_features_all['std_iat'].fillna(0)
flow_features_all['mean_iat']      = flow_features_all['mean_iat'].fillna(0)
flow_features_all['pkt_size_var']  = flow_features_all['pkt_size_var'].fillna(0)
flow_features_all['pkts_per_sec']  = flow_features_all['pkts_per_sec'].fillna(
    flow_features_all['packet_count'])
flow_features_all['bytes_per_sec'] = flow_features_all['bytes_per_sec'].fillna(
    flow_features_all['total_bytes'])

print(f"Total flows:    {len(flow_features_all):,}")
print(f"Total features: {len(flow_features_all.columns):,}")
print()
print("Protocol groups:")
print(flow_features_all['protocol_group'].value_counts())
print()
print("Class distribution:")
print(f"  Normal (0):  {(flow_features_all['is_anomaly']==0).sum():,}")
print(f"  Anomaly (1): {(flow_features_all['is_anomaly']==1).sum():,}")

# Standardisation + SVD

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
import pickle

exclude_cols = [
    'Source', 'Destination', 'Protocol', 'protocol_group',
    'src_port_filled', 'dst_port_filled',
    'flow_start', 'flow_end', 'is_anomaly',
    'well_known_dst', 'well_known_src',
    'proto_ICMP', 'proto_TCP', 'proto_UDP',
    'protocol_encoded',
]

feature_cols_all = [
    c for c in flow_features_all.columns
    if c not in exclude_cols
    and pd.api.types.is_numeric_dtype(flow_features_all[c])
]

# drop zero/near-zero variance
selector = VarianceThreshold(threshold=0)
selector.fit(flow_features_all[feature_cols_all])
feature_cols_all = [c for c, keep in zip(feature_cols_all, selector.get_support()) if keep]

_X_temp      = StandardScaler().fit_transform(flow_features_all[feature_cols_all])
low_std      = [col for col, std in zip(feature_cols_all, _X_temp.std(axis=0)) if std < 0.01]
feature_cols_all = [c for c in feature_cols_all if c not in low_std]

X_all        = flow_features_all[feature_cols_all].values
y_all        = flow_features_all['is_anomaly'].values

scaler_all   = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_all)

print(f"Features: {len(feature_cols_all)}")
print(f"X shape:  {X_all_scaled.shape}")
print(f"y shape:  {y_all.shape}")
print()
print("Class distribution:")
print(f"  Normal (0):  {(y_all == 0).sum():,} ({(y_all == 0).mean()*100:.2f}%)")
print(f"  Anomaly (1): {(y_all == 1).sum():,} ({(y_all == 1).mean()*100:.2f}%)")

# SVD
svd_full = TruncatedSVD(n_components=min(21, len(feature_cols_all)-1), random_state=42)
svd_full.fit(X_all_scaled)
cumvar = np.cumsum(svd_full.explained_variance_ratio_)
n_components_all = int(np.argmax(cumvar >= 0.95)) + 1
print()
print(f"SVD components for 95% variance: {n_components_all}")
print(f"Cumulative variance:             {cumvar[n_components_all-1]*100:.2f}%")

svd_all   = TruncatedSVD(n_components=n_components_all, random_state=42)
X_all_svd = svd_all.fit_transform(X_all_scaled)
print(f"Reduced shape: {X_all_svd.shape}")

# Train and Compare Models

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, ConfusionMatrixDisplay, roc_curve,
                              precision_score, recall_score, f1_score)
import time

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all_svd, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}")
print()

# SVM
print("Training SVM...")
t0  = time.time()
svm_all = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
svm_all.fit(X_tr, y_tr)
svm_all_time = time.time() - t0
y_pred_svm_all = svm_all.predict(X_te)
y_prob_svm_all = svm_all.predict_proba(X_te)[:, 1]
auc_svm_all    = roc_auc_score(y_te, y_prob_svm_all)
print(f"Training time: {svm_all_time:.2f}s")
print()
print(" SVM RESULTS ( 3M dataset)")
print(classification_report(y_te, y_pred_svm_all, target_names=['Normal', 'Anomaly']))
print(f"ROC AUC: {auc_svm_all:.4f}")
print()

# Naive Bayes
print("Training Naive Bayes...")
t0  = time.time()
nb_all = GaussianNB()
nb_all.fit(X_tr, y_tr)
nb_all_time = time.time() - t0
y_pred_nb_all = nb_all.predict(X_te)
y_prob_nb_all = nb_all.predict_proba(X_te)[:, 1]
auc_nb_all    = roc_auc_score(y_te, y_prob_nb_all)
print(f"Training time: {nb_all_time:.2f}s")
print()
print(" NAIVE BAYES RESULTS (3M dataset) ")
print(classification_report(y_te, y_pred_nb_all, target_names=['Normal', 'Anomaly']))
print(f"ROC AUC: {auc_nb_all:.4f}")

# Compare Single vs Multi-File

print(" SINGLE FILE vs MULTI-FILE COMPARISON ")
print()
print(f"{'Metric':<20} {'SVM (1 file)':>14} {'SVM (2 files)':>14} {'NB (1 file)':>12} {'NB (2 files)':>12}")
print("-" * 75)

metrics = {
    'Precision': [
        precision_score(y_test, y_pred_svm),
        precision_score(y_te,   y_pred_svm_all),
        precision_score(y_test, y_pred_nb),
        precision_score(y_te,   y_pred_nb_all),
    ],
    'Recall': [
        recall_score(y_test, y_pred_svm),
        recall_score(y_te,   y_pred_svm_all),
        recall_score(y_test, y_pred_nb),
        recall_score(y_te,   y_pred_nb_all),
    ],
    'F1 Score': [
        f1_score(y_test, y_pred_svm),
        f1_score(y_te,   y_pred_svm_all),
        f1_score(y_test, y_pred_nb),
        f1_score(y_te,   y_pred_nb_all),
    ],
    'ROC AUC': [auc_svm, auc_svm_all, auc_nb, auc_nb_all],
}

for metric, values in metrics.items():
    print(f"{metric:<20} {values[0]:>14.4f} {values[1]:>14.4f} {values[2]:>12.4f} {values[3]:>12.4f}")

