//+------------------------------------------------------------------+
//|                                          Demo_FileReadStruct.mq5 |
//|                        Copyright 2013, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2013, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 20
#property indicator_plots 6

#property indicator_label1 "std+1"
#property indicator_color1 clrGreen
#property indicator_type1 DRAW_LINE
#property indicator_width1 1

#property indicator_label2 "std+2"
#property indicator_color2 clrGreen
#property indicator_type2 DRAW_LINE
#property indicator_width2 1

#property indicator_label3 "std+3"
#property indicator_color3 clrGreen
#property indicator_type3 DRAW_LINE
#property indicator_width3 1

#property indicator_label4 "std-1"
#property indicator_color4 clrRed
#property indicator_type4 DRAW_LINE
#property indicator_width4 1

#property indicator_label5 "std-2"
#property indicator_color5 clrRed
#property indicator_type5 DRAW_LINE
#property indicator_width5 1

#property indicator_label6 "std-3"
#property indicator_color6 clrRed
#property indicator_type6 DRAW_LINE
#property indicator_width6 1


#include <Math\Stat\Math.mqh>



//--- parameters for receiving data
input bool     Show_Current_std_only = true;//Only Show Current STD
input bool     InpExtendPanel = true;//Extend Panel

input string   InpMomentIndic = "atr";//"std" OR "atr"

input color    InpMainSymbolColor = clrWhite;//Main Symbol Color
input color    InpFontColor = clrWhite;//Font Color
input color    InpBGColor = clrDimGray;//Background Color
input color    InpBorderColor = clrBlack;//Border Color

//+------------------------------------------------------------------+
//| Structure for storing candlestick data                           |
//+------------------------------------------------------------------+
struct Table
  {
   double            corr;
   double            symbol_id;
   double            corr_direct;
   double            oc;
   double            hl;
   double            chng_direct;
  };

string symbols_buff[];
string SymbolsBuffer[];



//--- indicator handles
int std_handle;
int atr_handle;
int ma_handle;


//--- indicator buffers
double std_buff[];
double atr_buff[];
double ma_buff[];
double stdp1_buff[];
double stdp2_buff[];
double stdp3_buff[];
double stdm1_buff[];
double stdm2_buff[];
double stdm3_buff[];


double atrp1_buff[];
double atrp2_buff[];
double atrp3_buff[];
double atrm1_buff[];
double atrm2_buff[];
double atrm3_buff[];

//--- global variables
Table table_buff[];
int          size=0;
int          ind=0;

string  InpFileName1="Correlation.bin"; // file name
string  InpDirectoryName="Data";  // directory name

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int heightScreen=ChartGetInteger(ChartID(),CHART_HEIGHT_IN_PIXELS,0);
int widthScreen=ChartGetInteger(ChartID(),CHART_WIDTH_IN_PIXELS,0);
int default_size=100;

int Num_Symbols = 46;
color clr;
color txt_clr;

bool show_data = true;
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {

   SetIndexBuffer(ChartID(),stdp1_buff,INDICATOR_DATA);
   SetIndexBuffer(1,stdp2_buff,INDICATOR_DATA);
   SetIndexBuffer(2,stdp3_buff,INDICATOR_DATA);
   SetIndexBuffer(3,stdm1_buff,INDICATOR_DATA);
   SetIndexBuffer(4,stdm2_buff,INDICATOR_DATA);
   SetIndexBuffer(5,stdm3_buff,INDICATOR_DATA);
   ArraySetAsSeries(stdp1_buff,true);
   ArraySetAsSeries(stdp2_buff,true);
   ArraySetAsSeries(stdp3_buff,true);
   ArraySetAsSeries(stdm1_buff,true);
   ArraySetAsSeries(stdm2_buff,true);
   ArraySetAsSeries(stdm3_buff,true);


   SetIndexBuffer(ChartID(),atrp1_buff,INDICATOR_DATA);
   SetIndexBuffer(1,atrp2_buff,INDICATOR_DATA);
   SetIndexBuffer(2,atrp3_buff,INDICATOR_DATA);
   SetIndexBuffer(3,atrm1_buff,INDICATOR_DATA);
   SetIndexBuffer(4,atrm2_buff,INDICATOR_DATA);
   SetIndexBuffer(5,atrm3_buff,INDICATOR_DATA);
   ArraySetAsSeries(atrp1_buff,true);
   ArraySetAsSeries(atrp2_buff,true);
   ArraySetAsSeries(atrp3_buff,true);
   ArraySetAsSeries(atrm1_buff,true);
   ArraySetAsSeries(atrm2_buff,true);
   ArraySetAsSeries(atrm3_buff,true);

   std_handle = iStdDev(_Symbol,PERIOD_CURRENT,75,0,MODE_SMA,PRICE_OPEN);
   atr_handle = iATR(_Symbol,PERIOD_CURRENT,75);
   ma_handle = iMA(_Symbol,PERIOD_CURRENT,75,0,MODE_SMA,PRICE_OPEN);


// createButton("show_hide","Hide",30,20,775,0,"Arial",7,InpBGColor,InpBorderColor,InpFontColor);


   string InpFileName = _Symbol+"_"+InpFileName1;

   size = 0;

   ArrayResize(table_buff,default_size);
   ArrayResize(SymbolsBuffer,SymbolsTotal(false));
//--- open the file
   ResetLastError();
   int file_handle=FileOpen(InpDirectoryName+"//"+InpFileName,FILE_READ|FILE_BIN);
   int file_handle1=FileOpen(InpDirectoryName+"//"+"symbols_list.txt",FILE_READ|FILE_TXT|FILE_ANSI);
   if(file_handle!=INVALID_HANDLE && file_handle1!=INVALID_HANDLE)
     {
      PrintFormat("%s file is available for reading",InpFileName, "Symbols_list");
      PrintFormat("File path: %s\\Files\\",TerminalInfoString(TERMINAL_COMMONDATA_PATH));
      //--- read data from the file
      while(!FileIsEnding(file_handle))
        {
         //--- write data to the array
         FileReadStruct(file_handle,table_buff[size]);
         size++;
         //--- check if the array is overflown
         if(size==default_size)
           {
            //--- increase the array size
            default_size+=100;
            ArrayResize(table_buff,default_size);
           }
        }

      //--- close the file
      FileClose(file_handle);
      PrintFormat("Data is read, %s file is closed",InpFileName);


      int i = 0;
      //FileReadArray(file_handle1,symbols_buff);
      while(!FileIsEnding(file_handle1))
        {
         SymbolsBuffer[i] = FileReadString(file_handle1);
         i++;
        }

      //--- close the file
      FileClose(file_handle1);
      PrintFormat("Data is read, %s file is closed","symbols_list.bin");

     }
   else
     {
      PrintFormat("failed to open 1 %s file, Error code = %d",InpFileName,GetLastError());
      return(INIT_FAILED);
     }

   string txt = "";
   /**
      for(int i=0; i<ArraySize(symbols_buff)-1; i++)
        {
         if(symbols_buff[i]!="")
            txt += symbols_buff[i];


         if(symbols_buff[i]!="" && symbols_buff[i+1]=="" || i==ArraySize(symbols_buff) || (txt=="BADGERUSD" || txt=="PUNDIXUSD"))
           {
            ArrayResize(SymbolsBuffer, ArraySize(SymbolsBuffer)+1);
            SymbolsBuffer[ArraySize(SymbolsBuffer)-1] =  txt;

            Print(ArraySize(SymbolsBuffer)-1, txt);
            txt = "";
           }

        }
   **/


   int h = 20;
   int padding = 40;
   int y = padding - h;


   heightScreen=ChartGetInteger(ChartID(),CHART_HEIGHT_IN_PIXELS,0);
   widthScreen=ChartGetInteger(ChartID(),CHART_WIDTH_IN_PIXELS,0);

   Num_Symbols = heightScreen/h-3;




   int header_y = y;
   int header_x1 = 0;
   int header_x2 = 100;
   int header_x3 = 150;
   int header_x4 = 200;

   if(InpExtendPanel)
     {
      header_y = 0;
      header_x1 = 420;
      header_x2 = 520;
      header_x3 = 570;
      header_x4 = 620;
     }

   createButton("btn_header1", "Symbol", 100,h,header_x1,header_y,"Arial", 12,InpBGColor, clrBlack,InpFontColor);
   createButton("btn_value_header1","Corr", 50,h,header_x2,header_y,"Arial", 10,InpBGColor, clrBlack,InpFontColor);
   createButton("btn_oc_header1","Delta", 50,h,header_x3,header_y,"Arial", 10,InpBGColor, clrBlack,InpFontColor);
//createButton("btn_hl_header1","H-L %", 50,h,header_x4,header_y,"Arial", 10,InpBGColor, clrBlack,InpFontColor);




   if(InpExtendPanel)
      Num_Symbols *= 3;

   for(int i=Num_Symbols-1; i<1000; i++)
     {
      ObjectDelete(ChartID(),"btn_symbol_"+(string) i);
      ObjectDelete(ChartID(),"btn_value_"+(string) i);
      ObjectDelete(ChartID(),"btn_oc_"+(string) i);
      ObjectDelete(ChartID(),"btn_hl_"+(string) i);
     }

   for(int i=0; i<Num_Symbols-1; i++)
     {

      if(table_buff[i].oc<=1)
        {
         clr = clrLightSalmon;
         txt_clr = clrBrown;
        }
        else
          {
           clr = clrLimeGreen;
           txt_clr = clrGreen;
          }
      if(table_buff[i].corr_direct>0 && table_buff[i].oc>0)
        {
         clr = clrLightSalmon;
         txt_clr = clrMaroon;
        }
      if(table_buff[i].corr_direct<0 && table_buff[i].oc<=0)
        {
         clr = clrLimeGreen;
         txt_clr = clrDarkGreen;
        }
      if(table_buff[i].corr_direct<0 && table_buff[i].oc>1)
        {
         clr = clrGreen;
         txt_clr = clrDarkGreen;
        }
      Print(ArraySize(SymbolsBuffer));
      Print(DoubleToInteger(table_buff[i].symbol_id));
      string symb = SymbolsBuffer[DoubleToInteger(table_buff[i].symbol_id)];







      int x1 = 0;
      int x2 = 100;
      int x3 = 150;
      int x4 = 200;
      int table_height = 0;

      int symb_width = 100;
      int symb_x =x1;


      if(InpExtendPanel)
        {

         x1 = 420;
         symb_x = x1;
         x2 = 520;
         x3 = 570;
         x4 = 620;
         table_height = h;
        }


      if(InpExtendPanel && i>Num_Symbols/3+1)
        {
         x1 = 210;
         symb_x = x1;
         x2 = 310;
         x3 = 360;
         x4 = 410;
         table_height = (Num_Symbols/3+3) * h;
        }
      if(InpExtendPanel && i>Num_Symbols*2/3+3)
        {
         x1 = 0;
         symb_x = x1;
         x2 = 100;
         x3 = 150;
         x4 = 200;
         table_height = (Num_Symbols*2/3+5) * h;
        }

      if(i==0)
        {
         int star_size = 25;
         symb_width = 100 - star_size;
         symb_x += star_size;
         createButton("btn_symbol_main", "✪",star_size,h,symb_x-star_size,padding-table_height,"Arial Black", 13, InpBGColor, clrBlack, InpMainSymbolColor);
         createButton("btn_symbol_"+(string) i,symb, symb_width+100,h,symb_x,padding+i*h-table_height,"Arial Black", 10,InpBGColor, clrBlack,InpFontColor);
        }
      else
        {
         createButton("btn_symbol_"+(string) i,symb, symb_width,h,symb_x,padding+i*h-table_height,"Arial Black", 10,InpBGColor, clrBlack,InpFontColor);
         createButton("btn_value_"+(string) i,(string) NormalizeDouble(table_buff[i].corr*100,1)+"%", 50,h,x2,padding+i*h-table_height,"Arial", 10,table_buff[i].corr_direct>0? clrGreen : clrCrimson, clrBlack,clrWhite);
         createButton("btn_oc_"+(string) i,(string) NormalizeDouble(table_buff[i].oc,3), 50,h,x3,padding+i*h-table_height,"Arial Black", 10,clr,clrBlack,txt_clr);
         //createButton("btn_hl_"+(string) i,(string) NormalizeDouble(table_buff[i].hl,3), 50,h,x4,padding+i*h-table_height,"Arial Black", 10,clr, clrBlack,txt_clr);

        }



     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
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

   MqlCalendarValue values[];
   MqlDateTime now;
   TimeCurrent(now);

   if(MathMod(now.min, 5)==0)
     {
      Print("Updating Calendar");
      //---
      // Calendar Script
      //---


      //------------------- All events values ---------

      string EU_code="EU";

      datetime date_from = StringToTime(StringFormat("%02d.%02d.%4d",now.day,now.mon,now.year));
      datetime date_to=0;

      if(CalendarValueHistory(values,date_from,date_to))
        {

         int h=FileOpen("Data/All_Events_Values.csv",FILE_READ|FILE_WRITE|FILE_CSV);

         FileWrite(h, "id", "event_id", "time", "period", "revision", "actual_value", "prev_value", "revised_prev_value", "forecast_value", "impact_type");

         for(int i=0; i<ArraySize(values); i++)
           {
            FileWrite(h, values[i].id, values[i].event_id, values[i].time, values[i].period, values[i].revision, values[i].actual_value, values[i].prev_value, values[i].revised_prev_value, values[i].forecast_value, values[i].impact_type);

           }
         FileClose(h);

         //----------------------------------------------




         //--- 1) Extracting COUNTRIES LIST -----


         string Currencies[];

         MqlCalendarCountry countries[];
         int count=CalendarCountries(countries);

         if(count>0)
           {
            int curr_size = ArraySize(countries);
            ArrayResize(Currencies, curr_size);
            for(int i=0; i<curr_size; i++)
              {
               Currencies[i] = countries[i].currency;
              }

            h=FileOpen("Data/Countries.csv",FILE_READ|FILE_WRITE|FILE_CSV);

            FileWrite(h, "id", "name", "currency");

            for(int j=0; j<ArraySize(countries); j++)
              {
               FileWrite(h, countries[j].id, countries[j].name, countries[j].currency);
              }
            FileClose(h);

           }
         else
           {
            Print("Failed to receive currencies, error code : %d", GetLastError());
           }

         //---------------------------------



         //----- 2) Extract Events By Currency -------

         MqlCalendarEvent events[];
         //--- get EU currency events
         for(int i=0; i<ArraySize(Currencies); i++)
           {

            string symbol = Currencies[i];

            int count1 = CalendarEventByCurrency(symbol,events);

            h=FileOpen("Data/Events_"+symbol+".csv",FILE_READ|FILE_WRITE|FILE_CSV);

            FileWrite(h, "id", "type", "frequency", "time_mode", "country_id", "unit", "importance", "multiplier", "digits", "source_url", "event_code", "name");

            for(int j=0; j<ArraySize(events); j++)
              {
               FileWrite(h, events[j].id, events[j].type, events[j].frequency, events[j].time_mode, events[j].country_id, events[j].unit, events[j].importance, events[j].multiplier, events[j].digits, events[j].source_url, events[j].event_code, events[j].name);

              }
            FileClose(h);


           }


        }
      else
        {
         PrintFormat("Error! Failed to receive events for country_code=%s",EU_code);
         PrintFormat("Error code: %d",GetLastError());
        }

      //--- End of Calendar Script


     }





   ArraySetAsSeries(std_buff,true);
   ArraySetAsSeries(atr_buff,true);
   ArraySetAsSeries(ma_buff,true);

   CopyBuffer(std_handle,0,0,rates_total,std_buff);
   CopyBuffer(atr_handle,0,0,rates_total,atr_buff);
   CopyBuffer(ma_handle,0,0,rates_total,ma_buff);

   OnInit();

   int total_std = rates_total-1;
   int init_value = prev_calculated;

   if(Show_Current_std_only)
      total_std = 1;
   init_value = 0;


   int std_height = 2;


   for(int i=init_value; i<total_std ; i++)
     {

      double stdp1;
      double stdp2;
      double stdp3;
      double stdm1;
      double stdm2;
      double stdm3;

      if(InpMomentIndic=="atr")
        {

         stdp1 = open[rates_total-i-1] + atr_buff[i];
         stdp2 = open[rates_total-i-1] + atr_buff[i] * 2;
         stdp3 = open[rates_total-i-1] + atr_buff[i] * 3;
         stdm1 = open[rates_total-i-1] - atr_buff[i];
         stdm2 = open[rates_total-i-1] - atr_buff[i] * 2;
         stdm3 = open[rates_total-i-1] - atr_buff[i] * 3;
        }
      else
         if(InpMomentIndic=="std")
           {

            stdp1 = open[rates_total-i-1] + atr_buff[i];
            stdp2 = open[rates_total-i-1] + atr_buff[i] * 2;
            stdp3 = open[rates_total-i-1] + atr_buff[i] * 3;
            stdm1 = open[rates_total-i-1] - atr_buff[i];
            stdm2 = open[rates_total-i-1] - atr_buff[i] * 2;
            stdm3 = open[rates_total-i-1] - atr_buff[i] * 3;
           }

      if(Show_Current_std_only)
        {

         int stdp1_x;
         int stdp1_y;
         double stdp1_price2;
         datetime stdp1_time2;

         int stdp2_x;
         int stdp2_y;
         double stdp2_price2;
         datetime stdp2_time2;


         int stdp3_x;
         int stdp3_y;
         double stdp3_price2;
         datetime stdp3_time2;




         int stdm1_x;
         int stdm1_y;
         double stdm1_price2;
         datetime stdm1_time2;


         int stdm2_x;
         int stdm2_y;
         double stdm2_price2;
         datetime stdm2_time2;


         int stdm3_x;
         int stdm3_y;
         double stdm3_price2;
         datetime stdm3_time2;
         int sub;

         ChartTimePriceToXY(ChartID(),0,time[rates_total-i-1],stdp1,stdp1_x,stdp1_y);
         ChartXYToTimePrice(ChartID(),stdp1_x,stdp1_y+std_height,sub,stdp1_time2,stdp1_price2);
         ChartTimePriceToXY(ChartID(),0,time[rates_total-i-3],stdp1,stdp1_x,stdp1_y);

         ChartTimePriceToXY(ChartID(),0,time[rates_total-i-1],stdp2,stdp2_x,stdp2_y);
         ChartXYToTimePrice(ChartID(),stdp2_x,stdp2_y+std_height,sub,stdp2_time2,stdp2_price2);
         ChartTimePriceToXY(ChartID(),0,time[rates_total-i-3],stdp2,stdp2_x,stdp2_y);

         ChartTimePriceToXY(ChartID(),0,time[rates_total-i-1],stdp3,stdp3_x,stdp3_y);
         ChartXYToTimePrice(ChartID(),stdp3_x,stdp3_y+std_height,sub,stdp3_time2,stdp3_price2);
         ChartTimePriceToXY(ChartID(),0,time[rates_total-i-3],stdp3,stdp3_x,stdp3_y);

         ChartTimePriceToXY(ChartID(),0,time[rates_total-i-1],stdm1,stdm1_x,stdm1_y);
         ChartXYToTimePrice(ChartID(),stdm1_x,stdm1_y+std_height,sub,stdm1_time2,stdm1_price2);
         ChartTimePriceToXY(ChartID(),0,time[rates_total-i-3],stdm1,stdm1_x,stdm1_y);

         ChartTimePriceToXY(ChartID(),0,time[rates_total-i-1],stdm2,stdm2_x,stdm2_y);
         ChartXYToTimePrice(ChartID(),stdm2_x,stdm2_y+std_height,sub,stdm2_time2,stdm2_price2);
         ChartTimePriceToXY(ChartID(),0,time[rates_total-i-3],stdm2,stdm2_x,stdm2_y);

         ChartTimePriceToXY(ChartID(),0,time[rates_total-i-1],stdm3,stdm3_x,stdm3_y);
         ChartXYToTimePrice(ChartID(),stdm3_x,stdm3_y+std_height,sub,stdm3_time2,stdm3_price2);
         ChartTimePriceToXY(ChartID(),0,time[rates_total-i-3],stdm3,stdm3_x,stdm3_y);






         createRectangle("STD+1",time[rates_total-i-1],stdp1,time[rates_total-i-2],stdp1_price2,clrLime);
         createLabel("STD+1_price", stdp1_x,stdp1_y-20,(string) NormalizeDouble(stdp1,_Digits),clrLime);
         createRectangle("STD+2",time[rates_total-i-1],stdp2,time[rates_total-i-2],stdp2_price2,clrGreen);
         createLabel("STD+2_price", stdp2_x,stdp2_y-20,(string) NormalizeDouble(stdp2,_Digits),clrGreen);
         createRectangle("STD+3",time[rates_total-i-1],stdp3,time[rates_total-i-2],stdp3_price2,clrDarkGreen);
         createLabel("STD+3_price", stdp3_x,stdp3_y-20,(string) NormalizeDouble(stdp3,_Digits),clrDarkGreen);

         createRectangle("STD-1",time[rates_total-i-1],stdm1,time[rates_total-i-2],stdm1_price2,clrTomato);
         createLabel("STD-1_price", stdm1_x,stdm1_y+3,(string) NormalizeDouble(stdm1,_Digits),clrTomato);
         createRectangle("STD-2",time[rates_total-i-1],stdm2,time[rates_total-i-2],stdm2_price2,clrCrimson);
         createLabel("STD-2_price", stdm2_x,stdm2_y+3,(string) NormalizeDouble(stdm2,_Digits),clrCrimson);
         createRectangle("STD-3",time[rates_total-i-1],stdm3,time[rates_total-i-2],stdm3_price2,clrDarkRed);
         createLabel("STD-3_price", stdm3_x,stdm3_y+3,(string) NormalizeDouble(stdm3,_Digits),clrDarkRed);


         //createPriceLine(InpMomentIndic+"+1_price",stdp1,clrLime);
         //createPriceLine(InpMomentIndic+"+2_price",stdp2,clrGreen);
         //createPriceLine(InpMomentIndic+"+3_price",stdp3,clrDarkGreen);
         //createPriceLine(InpMomentIndic+"-1_price",stdm1,clrTomato);
         //createPriceLine(InpMomentIndic+"-2_price",stdm2,clrCrimson);
         //createPriceLine(InpMomentIndic+"-3_price",stdm3,clrDarkRed);





        }
      else
        {

         ObjectDelete(ChartID(),"STD+1");
         ObjectDelete(ChartID(),"STD+2");
         ObjectDelete(ChartID(),"STD+3");
         ObjectDelete(ChartID(),"STD-1");
         ObjectDelete(ChartID(),"STD-2");
         ObjectDelete(ChartID(),"STD-3");

         stdp1_buff[i] = stdp1;
         stdp2_buff[i] = stdp2;
         stdp3_buff[i] = stdp3;
         stdm1_buff[i] = stdm1;
         stdm2_buff[i] = stdm2;
         stdm3_buff[i] = stdm3;

        }


     }

   string InpFileName = _Symbol+"_"+InpFileName1;

   size = 0;

   ArrayResize(table_buff,default_size);
//--- open the file
   ResetLastError();
   int file_handle=FileOpen(InpDirectoryName+"//"+InpFileName,FILE_READ|FILE_BIN|FILE_COMMON);
   int file_handle1=FileOpen(InpDirectoryName+"//"+"symbols_list.txt",FILE_READ|FILE_TXT|FILE_COMMON|FILE_ANSI, '\n');
   if(file_handle!=INVALID_HANDLE && file_handle1!=INVALID_HANDLE)
     {
      PrintFormat("%s file is available for reading",InpFileName, "Symbols_list");
      PrintFormat("File path: %s\\Files\\",TerminalInfoString(TERMINAL_COMMONDATA_PATH));
      //--- read data from the file
      while(!FileIsEnding(file_handle))
        {
         //--- write data to the array
         FileReadStruct(file_handle,table_buff[size]);
         size++;
         //--- check if the array is overflown
         if(size==default_size)
           {
            //--- increase the array size
            default_size+=100;
            ArrayResize(table_buff,default_size);
           }
        }

      //--- close the file
      FileClose(file_handle);
      PrintFormat("Data is read, %s file is closed",InpFileName);

      int i = 0;
      //FileReadArray(file_handle1,symbols_buff);
      while(!FileIsEnding(file_handle1))
        {
         SymbolsBuffer[i] = FileReadString(file_handle1);
         i++;
        }


      //FileReadArray(file_handle1,symbols_buff);

      //--- close the file
      FileClose(file_handle1);
      PrintFormat("Data is read, %s file is closed","symbols_list.bin");

     }
   else
     {
      PrintFormat("failed to open 2 %s file, Error code = %d",InpFileName,GetLastError());

     }





   for(int i=0; i<Num_Symbols; i++)
     {
      
      string symb = SymbolsBuffer[DoubleToInteger(table_buff[i].symbol_id)];
      double symb_oc = NormalizeDouble(GetOC_Chng(symb),3);
      double symb_hl = NormalizeDouble(GetHL_Chng(symb),3);
      color symb_color = ObjectGetInteger(ChartID(),"btn_symbol_"+(string) i,OBJPROP_COLOR);
      color symb_bgcolor = ObjectGetInteger(ChartID(),"btn_symbol_"+(string) i,OBJPROP_BGCOLOR);

      if(symb_oc<=0)
        {
         clr = clrLightSalmon;
         txt_clr = clrBrown;
        }
      if(symb_oc>0)
        {
         clr = clrLimeGreen;
         txt_clr = clrDarkGreen;
        }

      if(i==0)
         symb += "";


      datetime star_time;
      double star_price;
      int z = 0;

      ChartXYToTimePrice(ChartID(),400,100,z,star_time,star_price);

      ObjectCreate(ChartID(),"star",OBJ_BITMAP_LABEL,0,star_time,star_price);
      ObjectSetString(ChartID(),"star",OBJPROP_BMPFILE,"\\Image\\dollar.bmp");



      if(i==0)
        {
         symb_color = InpMainSymbolColor;
         modifyButton("btn_symbol_main","✪",symb_bgcolor,symb_color);
        }
      else
        {

         modifyButton("btn_symbol_"+(string) i,symb,symb_bgcolor,symb_color);
         modifyButton("btn_value_"+(string) i,(string) NormalizeDouble(table_buff[i].corr*100,1) + "%",table_buff[i].corr_direct>0? clrGreen : clrCrimson,clrWhite);
         modifyButton("btn_oc_"+(string) i,(string) symb_oc, clr,txt_clr);
         modifyButton("btn_hl_"+(string) i,(string) symb_hl, clr,txt_clr);


        }





      //createButton("btn_value_"+(string) i,(string) NormalizeDouble(table_buff[i].corr,2), 50,h,100,100+i*h,"Arial", 10,table_buff[i].corr_direct>0? clrGreen : clrCrimson, clrBlack,clrWhite);

      //createButton("btn_oc_"+(string) i,(string) NormalizeDouble(table_buff[i].oc,2), 50,h,150,100+i*h,"Arial", 10,clr,clrBlack,txt_clr);
      //createButton("btn_hl_"+(string) i,(string) NormalizeDouble(table_buff[i].hl,2), 50,h,200,100+i*h,"Arial", 10,clr, clrBlack,txt_clr);

     }


   double Spread[1000];

   for(int i=0; i<1000; i++)
     {
      Spread[i] = spread[i];
     }

   double avg_spread = MathMean(Spread);





   createButton("btn_ATR_","ATR", 100,25,900,0,"Arial Black", 10,InpBGColor, clrBlack,InpFontColor);
   createButton("btn_ATR_value_",(string) NormalizeDouble(atr_buff[0]*pow(10, _Digits-1),1), 100,25,900,25,"Arial Black", 10,InpBGColor, clrBlack,InpFontColor);

   createButton("btn_AvgSpread_","Avg Spread", 100,25,1000,0,"Arial Black", 10,InpBGColor, clrBlack,InpFontColor);
   createButton("btn_AvgSpread_value_",(string) avg_spread, 100,25,1000,25,"Arial Black", 10,InpBGColor, clrBlack,InpFontColor);

   createButton("btn_STD_","STD", 100,25,1100,0,"Arial Black", 10,InpBGColor, clrBlack,InpFontColor);
   createButton("btn_STD_value_",(string) NormalizeDouble(std_buff[0]*pow(10, _Digits-1),1), 100,25,1100,25,"Arial Black", 10,InpBGColor, clrBlack,InpFontColor);




   return(rates_total);
  }



//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,const long& lparam,const double& dparam,const string& sparam)
  {
   handleButtonClicks();
   OnInit();
   if(show_data)
     {
      ObjectSetInteger(ChartID(),"btn_symbol_0",OBJPROP_HIDDEN,100);
      ObjectSetString(ChartID(),"show_hide",OBJPROP_TEXT,"Show");


     }
   else
     {

      ObjectSetInteger(ChartID(),"btn_symbol_0",OBJPROP_YDISTANCE,100);
      ObjectSetString(ChartID(),"show_hide",OBJPROP_TEXT,"Hide");
     }
  }




//+------------------------------------------------------------------+
//|  Custom Functions                                                |
//+------------------------------------------------------------------+
int DoubleToInteger(double x)
  {
   return round(x);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void createRectangle(string rect_id, datetime date1, double price1, datetime date2, double price2, color bgColor)
  {
   ObjectDelete(ChartID(),rect_id);
   ObjectCreate(ChartID(), rect_id, OBJ_RECTANGLE,0,date1,price1,date2,price2);
   ObjectSetInteger(ChartID(),rect_id, OBJPROP_COLOR, bgColor);
   ObjectSetInteger(ChartID(),rect_id, OBJPROP_FILL, bgColor);
   ObjectSetString(ChartID(),rect_id, OBJPROP_TEXT, "H");

  }


//+------------------------------------------------------------------+
void createButton(string buttonID, string buttonText, int width, int height, int x, int y, string font, int fontSize, color bgColor, color borderColor, color txtColor)
  {
//ObjectDelete(ChartID(), buttonID);
   ObjectCreate(ChartID(), buttonID, OBJ_BUTTON, 0, 0, 0);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_COLOR, txtColor);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_BGCOLOR, bgColor);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_BORDER_COLOR, InpBorderColor);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_XSIZE, width);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_YSIZE, height);
   ObjectSetString(ChartID(), buttonID, OBJPROP_FONT, font);
   ObjectSetString(ChartID(), buttonID, OBJPROP_TEXT, buttonText);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_FONTSIZE, fontSize);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_SELECTABLE,0);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_XDISTANCE, 9999);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_YDISTANCE, 9999);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_ZORDER, 999999999999999999999999999999999);
   ObjectSetString(ChartID(), buttonID, OBJPROP_BMPFILE, "star.bmp");

  }
//+------------------------------------------------------------------+
void modifyButton(string buttonID, string buttonText, color bgColor, color txtColor)
  {
   ObjectSetString(ChartID(), buttonID, OBJPROP_TEXT, buttonText);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_COLOR, txtColor);
   ObjectSetInteger(ChartID(), buttonID, OBJPROP_BGCOLOR, bgColor);
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string AssembleName(const string &arr[], int index)
  {

   string txt = "";
   for(int i=index; i<5; i++)
     {
      Print(arr[i]);
      if(arr[i]!=" ")
         txt += arr[i];
      else
         return txt;
     }
   return txt;
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetOC_Chng(string symbol)
  {
   ENUM_TIMEFRAMES tf = PERIOD_CURRENT;

   return (iClose(symbol,tf,0) - iOpen(symbol,tf,0)) *100 / iOpen(symbol, tf, 0);
  }


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetHL_Chng(string symbol)
  {
   ENUM_TIMEFRAMES tf = PERIOD_CURRENT;

   return (iHigh(symbol,tf,0) - iLow(symbol,tf,0)) *100 / iLow(symbol, tf, 0);
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+

//+------------------------------------------------------------------------------------------------------------------+
void handleButtonClicks()
  {
   if(ObjectGetInteger(ChartID(), "show_hide", OBJPROP_STATE))
     {
      ObjectSetInteger(ChartID(), "show_hide", OBJPROP_STATE, false);
      show_data = !show_data;
      GlobalVariableSet("Correlation" + "_visibility", show_data ? 1.0 : 0.0);
     }
  }
//+------------------------------------------------------------------+

//-------------------------------------------------------------------+
void createPriceLine(string id, double price, color xclr)
  {

   MqlDateTime tm;
   ObjectCreate(ChartID(),id,OBJ_RECTANGLE_LABEL,0,0,price);
   ObjectSetInteger(ChartID(),id,OBJPROP_COLOR,xclr);
   ObjectSetInteger(ChartID(),id,OBJPROP_STYLE,STYLE_SOLID);
   ObjectSetInteger(ChartID(),id,OBJPROP_WIDTH,1);
   ObjectSetInteger(ChartID(),id,OBJPROP_ZORDER,0);
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void createLabel(string id, int x, int y, string text, color xclr, int size1=10)
  {

   ObjectCreate(ChartID(),id,OBJ_LABEL,0,0,0);
   ObjectSetInteger(ChartID(),id,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(ChartID(),id,OBJPROP_YDISTANCE,y);
   ObjectSetInteger(ChartID(),id,OBJPROP_ZORDER,99999999);
   ObjectSetInteger(ChartID(),id,OBJPROP_COLOR,xclr);
   ObjectSetString(ChartID(),id,OBJPROP_TEXT,text);
   ObjectSetInteger(ChartID(),id,OBJPROP_FONTSIZE,size1);

  }
//+------------------------------------------------------------------+
