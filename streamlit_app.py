"""IC Ion Chromatography Automated Analysis: local Streamlit interface."""
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ic_analysis import (calibration_group, csv_bytes, detect_peaks, fit,
                         integrate, invert, parse_chromatogram, predict, quantify)

TITLE = 'IC Ion Chromatography Automated Analysis'
MODELS = {'Linear': 'linear', 'Quadratic': 'quadratic',
          'Piecewise (0.001 M boundary)': 'piecewise'}


def analyze(trace):
    record = {**trace, 'peaks': [], 'peak': None, 'confirmed': False, 'revision': 0}
    try:
        record['peaks'] = detect_peaks(trace['xy'])
        record['peak'] = record['peaks'][0]
    except ValueError as exc:
        record['error'] = str(exc)
    return record


def ingest(files, category):
    records = st.session_state.setdefault(category, {})
    for file in files or []:
        content = file.getvalue()
        key = hashlib.sha256(file.name.encode() + b'\0' + content).hexdigest()
        if key in records:
            continue
        try:
            if len(content) > 20 * 1024 ** 2:
                raise ValueError('File exceeds 20 MB.')
            trace = parse_chromatogram(content.decode('utf-8-sig'), file.name)
            if category == 'standards':
                match = re.fullmatch(r'NaI-([0-9.eE+-]+)M(?:[-_].*)?\.(?:txt|csv)', file.name, re.I)
                if not match:
                    raise ValueError('Standard filename must be NaI-<concentration>M.txt (or .csv).')
                concentration = float(match[1])
                if not math.isfinite(concentration) or concentration <= 0:
                    raise ValueError('Standard concentration must be finite and positive.')
                if not trace['method'] or trace['volume'] is None:
                    raise ValueError('Standards require Instrument Method and Volume (µl) metadata.')
                trace['concentration'] = concentration
            records[key] = analyze(trace)
        except (ValueError, UnicodeError) as exc:
            st.error(f'{file.name}: {exc}')
    return records


def inspector(records):
    if not records:
        return
    st.subheader('Chromatogram and integration review')
    key = st.selectbox('Trace', list(records), format_func=lambda k: records[k]['name'])
    trace = records[key]
    xy, peak = trace['xy'], trace['peak']
    fig = go.Figure(go.Scatter(x=xy[:, 0], y=xy[:, 1], name='Raw signal'))
    if peak:
        fig.add_trace(go.Scatter(x=[peak['left'], peak['right']], y=peak['baseline'], name='Linear baseline'))
        for edge in ('left', 'right'):
            fig.add_vline(x=peak[edge], line_dash='dot')
        st.write(f"Net area: {peak['area']:.8g} µS·min; retention time: {peak['rt']:.6g} min")
        if peak.get('clipped'):
            st.warning('Peak may be truncated. Quantification is blocked until reviewed.')
    else:
        st.warning(trace.get('error', 'Select integration bounds.'))
    fig.update_layout(xaxis_title='Retention time / min', yaxis_title='Signal / µS')
    st.plotly_chart(fig, width='stretch')
    revision = trace['revision']
    if trace['peaks']:
        choice = st.selectbox('Candidate peak (review ion identity)', range(len(trace['peaks'])),
                              format_func=lambda i: f"Peak {i+1}: {trace['peaks'][i]['rt']:.5g} min")
        if st.button('Use selected peak'):
            trace.update(peak=trace['peaks'][choice], revision=revision+1)
            st.rerun()
    with st.form(f'bounds_{key}_{revision}'):
        left = st.number_input('Integration start / min', value=float(peak['left'] if peak else xy[0, 0]))
        right = st.number_input('Integration end / min', value=float(peak['right'] if peak else xy[-1, 0]))
        if st.form_submit_button('Update integration'):
            try:
                updated = integrate(xy, left, right)
                updated['clipped'] = left == xy[0, 0] or right == xy[-1, 0]
                trace.update(peak=updated, revision=revision+1)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with st.expander('Automatic peak search'):
        start = st.number_input('Search start / min', min_value=0., value=2.)
        end = st.number_input('Search end / min (0 = end of trace)', min_value=0., value=0.)
        if st.button('Detect peaks again'):
            try:
                peaks = detect_peaks(xy, (start, end if end else math.inf))
                trace.update(peaks=peaks, peak=peaks[0], revision=revision+1)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))


def main():
    st.set_page_config(page_title=TITLE, page_icon='🧪', layout='wide')
    st.title(TITLE)
    st.caption('Automatic positive-peak detection, integration and concentration analysis. NaI / iodide workflow.')
    st.info('No calibration data are included. Upload your own standards to start. Files are processed by the Streamlit server in session memory; a remote deployment receives uploaded files.')
    with st.sidebar:
        kind = MODELS[st.selectbox('Calibration model', list(MODELS))]
        dilution = st.number_input('Sample dilution factor', min_value=0.000001, value=1.)
        st.caption('Original concentration = measured concentration × dilution factor. Exported instrument dilution metadata are not applied automatically.')
        if st.button('Clear all session data'):
            st.session_state.clear()
            st.rerun()
    standard_files = st.file_uploader('Upload calibration standards', type=['txt', 'csv'], accept_multiple_files=True, key='standard_upload')
    st.caption('Names: NaI-<concentration>M.txt or .csv. Include tab-separated Instrument Method and Volume (µl) metadata before a Time (min) header. Use matching units: min and µS.')
    standards = ingest(standard_files, 'standards')
    sample_files = st.file_uploader('Upload sample chromatograms', type=['txt', 'csv'], accept_multiple_files=True, key='sample_upload')
    samples = ingest(sample_files, 'samples')
    st.caption('Records and manual bounds persist in this session even if files are removed from an uploader. Download results before closing the session; use Clear all session data to reset.')
    model, reference, selected = None, None, []
    if standards:
        try:
            group = calibration_group(list(standards.values()))
            reference = group[0]
            st.caption(f"Calibration method: {reference['method']}; injection: {reference['volume']:g} µL. Only the largest matching group is available for fitting.")
            names = [t['name'] for t in group]
            if len(set(names)) != len(names):
                raise ValueError('Duplicate standard filenames: rename distinct replicates with a suffix, or clear the session and upload the intended files.')
            selected_names = st.multiselect('Included standards', names, default=names)
            selected = [t for t in group if t['name'] in selected_names and t['peak'] and not t['peak'].get('clipped')]
            if len(selected) < len(selected_names):
                st.warning('Standards without a complete peak are excluded. Review their integration.')
            model = fit([{**t, 'area': t['peak']['area']} for t in selected], kind)
        except ValueError as exc:
            st.error(str(exc))
    if model:
        st.subheader('Calibration')
        st.write(f"R²: {model['r2']:.6f}; area RMSE: {model['rmse']:.6g} µS·min; measured range: {model['measuredMin']:g}–{model['max']:g} M")
        x = np.linspace(model['min'], model['max'], 500)
        fig = go.Figure(go.Scatter(x=x, y=predict(model, x), name='Fitted curve'))
        fig.add_trace(go.Scatter(x=[t['concentration'] for t in selected], y=[t['peak']['area'] for t in selected], mode='markers', name='Measured standards'))
        fig.update_layout(xaxis_title='Concentration / M', yaxis_title='Net area / µS·min')
        st.plotly_chart(fig, width='stretch')
        rows = []
        for t in selected:
            try:
                bias = (invert(model, t['peak']['area'])['concentration']/t['concentration']-1)*100
            except ValueError:
                bias = None
            rows.append({'file': t['name'], 'concentration_M': t['concentration'], 'area_uS_min': t['peak']['area'], 'bias_percent': bias})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if any(r['bias_percent'] is not None and abs(r['bias_percent']) > 15 for r in rows):
            st.warning('Standard back-calculation bias exceeds 15%. Review calibration; this is a review flag, not a validated acceptance criterion.')
        if kind == 'piecewise':
            st.caption('The origin is a mathematical constraint, not a measured blank. Segments join at the measured 0.001 M boundary; the origin is excluded from R²/RMSE.')
        st.subheader('Sample results')
        results = []
        for key, trace in samples.items():
            if not trace['method'] or trace['volume'] is None:
                trace['confirmed'] = st.checkbox(f"{trace['name']}: confirm method {reference['method']}, injection {reference['volume']:g} µL, min and µS units", key=f'confirm_{key}')
            row = {'file': trace['name']}
            try:
                result = quantify(trace, trace['peak'], model, reference, dilution, trace['confirmed'])
                row.update(result)
                row['status'] = 'In measured range' if result['inRange'] else 'Below measured range' if result['belowMeasuredRange'] else 'Extrapolated'
                if len(trace['peaks']) > 1:
                    row['status'] += '; multiple peaks: review iodide identity'
            except ValueError as exc:
                row['status'] = str(exc)
            results.append(row)
        if results:
            st.dataframe(pd.DataFrame(results), hide_index=True)
        export = []
        for category, records in [('standard', standards.values()), ('sample', samples.values())]:
            for trace in records:
                peak = trace['peak'] or {}
                result_row = next((r for r in results if r['file'] == trace['name']), {}) if category == 'sample' else {}
                export.append({'type': category, 'file': trace['name'], 'standard_M': trace.get('concentration', ''),
                               'included': category == 'standard' and any(t is trace for t in selected),
                               'method': trace['method'], 'volume_uL': trace['volume'],
                               'left_min': peak.get('left', ''), 'right_min': peak.get('right', ''),
                               'area_uS_min': peak.get('area', ''), 'clipped': peak.get('clipped', ''),
                               'measured_M': result_row.get('concentration', ''), 'original_M': result_row.get('original_M', ''),
                               'dilution': dilution if category == 'sample' else '', 'segment': result_row.get('segment', ''),
                               'status': result_row.get('status', ''), 'model_parameters': json.dumps(model)})
        st.download_button('Download analysis CSV', csv_bytes(export), 'ic-analysis.csv', 'text/csv')
        st.download_button('Download model JSON', json.dumps(model, indent=2), 'ic-model.json', 'application/json')
        with st.expander('Model parameters'):
            st.json(model)
    inspector({**{'standard:'+k: v for k, v in standards.items()}, **{'sample:'+k: v for k, v in samples.items()}})
    with st.expander('Methods and limitations'):
        st.markdown('''- Positive peaks are located with a moving average and local noise thresholds. Integration uses unsmoothed data, a straight baseline between endpoints and trapezoidal integration with actual time steps.
- Review ion identity and integration bounds. The largest positive peak is not necessarily iodide. Standards and samples must share instrument method and injection volume.
- Piecewise fitting uses a fixed 0.001 M join, an origin-constrained quadratic low segment and a continuous linear high segment. Linear and quadratic alternatives are available.
- Results below or above measured standards are flagged. No LOD, LOQ or uncertainty is estimated without experimental validation.
- The concentration workflow assumes NaI / iodide (1:1); mass conversions use 149.894 and 126.90447 g/mol. Other ions require adapting the chemistry and validating the method.
- Uploaded data are kept in session memory. No sample files are intentionally written to disk or sent to external analysis services. Hosting infrastructure may retain its own logs.''')


if __name__ == '__main__':
    main()
