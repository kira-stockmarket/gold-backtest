def generate_excel_report(df, cluster_summary):
    print("📊 Generating Dynamic ML Backtest Report...")
    excel_df = pd.DataFrame()
    
    excel_df['Event_ID'] = [f"EVT_{i+1:03d}" for i in range(len(df))]
    excel_df['Event_Name'] = df['Year'].apply(lambda y: f"Diwali_{y}")
    excel_df['T0_Date'] = pd.to_datetime(df['Diwali_Date']).dt.strftime('%d-%m-%Y')
    
    excel_df['Assigned_Cluster'] = df['Regime_Cluster']
    excel_df['Train_Clus'] = df['Regime_Cluster'] 
    
    # Features as decimals
    excel_df['Feature_E'] = (df['Buildup_Return_%'] / 100).round(4)
    excel_df['Feature_P'] = (df['Pre_Diwali_15d_%'] / 100).round(4)
    
    def get_trend(val):
        return 'uptrend' if val > 0 else 'downtrend'

    # ---------------------------------------------------------
    # NEW DYNAMIC PRE-DIWALI PREDICTION LOGIC
    # ---------------------------------------------------------
    # Instead of blindly guessing "uptrend", we predict based on exhaustion.
    # If Feature_E (T-45 to T-15) exceeds an extreme upper bound (e.g., > 10% surge), 
    # the market is historically overbought, and we predict a 'downtrend' (profit taking).
    # Otherwise, seasonal festive demand kicks in, and we predict an 'uptrend'.
    
    exhaustion_threshold = 0.10  # 10% rally in 30 days indicates overbought conditions
    
    excel_df['Pred_Pre_Trend'] = excel_df['Feature_E'].apply(
        lambda x: 'downtrend' if x > exhaustion_threshold else 'uptrend'
    )
    
    excel_df['Actual_Pre_Ret'] = excel_df['Feature_P']
    excel_df['Actual_Pre_Trend'] = excel_df['Actual_Pre_Ret'].apply(get_trend)
    excel_df['Pre_Success'] = excel_df['Pred_Pre_Trend'] == excel_df['Actual_Pre_Trend']

    # ---------------------------------------------------------
    # POST-DIWALI PREDICTION
    # ---------------------------------------------------------
    cluster_means = cluster_summary.set_index('Regime_Cluster')['Avg_Post15']
    excel_df['Pred_Post_Trend'] = df['Regime_Cluster'].map(lambda c: get_trend(cluster_means[c]))
    
    excel_df['Actual_Post_Ret'] = (df['Post_Diwali_15d_%'] / 100).round(4)
    excel_df['Actual_Post_Trend'] = excel_df['Actual_Post_Ret'].apply(get_trend)
    excel_df['Post_Success'] = excel_df['Pred_Post_Trend'] == excel_df['Actual_Post_Trend']

    # --- Excel Formatting and Export ---
    wb = Workbook()
    ws = wb.active
    ws.title = "ML_Regime_Backtest"

    header_fill = PatternFill(start_color="34495E", end_color="34495E", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    align_center = Alignment(horizontal="center", vertical="center")
    thin_border = Border(left=Side(style='thin', color='BDC3C7'), right=Side(style='thin', color='BDC3C7'),
                         top=Side(style='thin', color='BDC3C7'), bottom=Side(style='thin', color='BDC3C7'))

    headers = ['Event_ID', 'Event_Name', 'T0_Date', 'Assigned_', 'Train_Clus', 'Feature_E', 'Feature_P', 
               'Pred_Pre_', 'Actual_Pre', 'Actual_Pre', 'Pre_Secto', 'Pred_Post', 'Actual_Po', 'Actual_Po', 'Post_Secto']
    
    ws.append(headers)
    for col in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = header_fill; cell.font = header_font; cell.alignment = align_center; cell.border = thin_border

    for r in dataframe_to_rows(excel_df, index=False, header=False):
        ws.append(r)

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=len(headers)):
        for cell in row:
            cell.border = thin_border; cell.alignment = align_center
            if cell.column in [6, 7, 9, 13]: cell.number_format = '0.0000'
            
            # True/False Validation Colors
            if cell.column in [11, 15]:
                cell.font = Font(color="27AE60", bold=True) if cell.value == True else Font(color="C0392B", bold=True)
            
            # Uptrend/Downtrend String Colors
            if cell.column in [8, 10, 12, 14]:
                if cell.value == 'uptrend': cell.font = Font(color="27AE60")
                elif cell.value == 'downtrend': cell.font = Font(color="C0392B")

    for col in ws.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length: max_length = len(str(cell.value))
            except: pass
        ws.column_dimensions[column].width = max_length + 3
    ws.freeze_panes = "A2"

    excel_path = os.path.join(OUTPUT_DIR, "gold_ml_backtest.xlsx")
    wb.save(excel_path)
    print(f"✅ Dynamic Excel saved to {excel_path}")
