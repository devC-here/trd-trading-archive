#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots   2

#property indicator_label1  "Upper Forecast"
#property indicator_type1   DRAW_SECTION
#property indicator_color1  clrLime
#property indicator_width1  2

#property indicator_label2  "Lower Forecast"
#property indicator_type2   DRAW_SECTION
#property indicator_color2  clrTomato
#property indicator_width2  2

input int    REF_CANDLES = 6;
input double EPS         = 1e-8;

double Upper[];
double Lower[];

//----------------------------------
bool IsRestricted(datetime t)
{
   MqlDateTime ts; TimeToStruct(t, ts);

   if(ts.hour <= 1) return true;
   if(ts.hour == 12 && ts.min >= 30) return true;
   if(ts.hour == 13) return true;
   if(ts.hour >= 18 && ts.hour <= 19) return true;

   return false;
}
//----------------------------------
double CandleSize(double o,double c)
{
   return MathAbs(o-c);
}
//----------------------------------
double VariationWeighted(double &arr[])
{
   int n=ArraySize(arr);
   if(n==0) return 0;

   double mean=0;
   for(int i=0;i<n;i++) mean+=arr[i];
   mean/=n;

   double num=0,den=0;
   for(int i=0;i<n;i++)
   {
      double w=1.0/(MathAbs(arr[i]-mean)+EPS);
      num+=arr[i]*w;
      den+=w;
   }

   return (den==0)?0:num/den;
}
//----------------------------------
double PredictForBar(int index,
                     const datetime &time[],
                     const double &open[],
                     const double &close[],
                     int rates_total)
{
   double sizes[];
   ArrayResize(sizes,0);

   int collected=0;

   for(int shift=index+1; shift<rates_total && collected<REF_CANDLES; shift++)
   {
      if(IsRestricted(time[shift]))
         continue;

      int n=ArraySize(sizes);
      ArrayResize(sizes,n+1);
      sizes[n]=CandleSize(open[shift],close[shift]);

      collected++;
   }

   if(collected<REF_CANDLES)
      return 0;

   return VariationWeighted(sizes);
}
//----------------------------------
int OnInit()
{
   SetIndexBuffer(0,Upper,INDICATOR_DATA);
   SetIndexBuffer(1,Lower,INDICATOR_DATA);

   ArraySetAsSeries(Upper,true);
   ArraySetAsSeries(Lower,true);

   IndicatorSetString(INDICATOR_SHORTNAME,"SLTP Per-Candle Projection");

   return(INIT_SUCCEEDED);
}
//----------------------------------
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   ArraySetAsSeries(time,true);
   ArraySetAsSeries(open,true);
   ArraySetAsSeries(close,true);

   int start = (prev_calculated==0) ? rates_total-20 : rates_total-prev_calculated;
   if(start<0) start=0;

   for(int i=start; i>=0; i--)
   {
      // clear first so sections don't connect
      Upper[i]=EMPTY_VALUE;
      Lower[i]=EMPTY_VALUE;

      double pred = PredictForBar(i,time,open,close,rates_total);
      if(pred<=0) continue;

      // draw ONLY this bar's projection
      Upper[i]=open[i]+pred;
      Lower[i]=open[i]-pred;
   }

   return rates_total;
}
