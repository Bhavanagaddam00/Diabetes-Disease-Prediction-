package com.example.diabetesdiseaseprediction;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.basgeekball.awesomevalidation.AwesomeValidation;
import com.basgeekball.awesomevalidation.ValidationStyle;
import com.google.common.collect.Range;

import org.json.JSONException;
import org.json.JSONObject;

public class MainActivity2 extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    EditText e1,e2;
    Button s;
    Spinner s1,s2,s3,s4,s5,s6;
    AwesomeValidation awesomeValidation;
    String pregnancies,glucose,bp,skinthickness,insulin,bmi,DPF,age;
    static int i=0;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);
        e1=findViewById(R.id.e1);
        e2=findViewById(R.id.e2);
        s=findViewById(R.id.s);
        s1=findViewById(R.id.spinner1);
        s2=findViewById(R.id.spinner2);
        s3=findViewById(R.id.spinner3);
        s4=findViewById(R.id.spinner4);
        s5=findViewById(R.id.spinner5);
        s6=findViewById(R.id.spinner6);


        awesomeValidation = new AwesomeValidation(ValidationStyle.BASIC);
        awesomeValidation.addValidation(this, R.id.e1, "[0-9]+",R.string.pregerr);
        awesomeValidation.addValidation(this, R.id.e2, Range.closed(2,30),R.string.sterr);

        Spinner dropdown = findViewById(R.id.spinner1);
        String[] items = new String[]{"low", "medium", "high"};
        Spinner dropdown1 = findViewById(R.id.spinner2);
        String[] items1 = new String[]{"low", "medium", "high"};
        Spinner dropdown2 = findViewById(R.id.spinner3);
        String[] items2 = new String[]{"low", "medium", "high"};
        Spinner dropdown3 = findViewById(R.id.spinner4);
        String[] items3 = new String[]{"low", "medium", "high"};
        Spinner dropdown4 = findViewById(R.id.spinner5);
        String[] items4 = new String[]{"normal", "prediabetes", "diabetes"};
        Spinner dropdown5 = findViewById(R.id.spinner6);
        String[] items5 = new String[]{"young", "middle", "old"};
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, items);
        dropdown.setAdapter(adapter);
        ArrayAdapter<String> adapter1 = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, items1);
        dropdown1.setAdapter(adapter1);
        ArrayAdapter<String> adapter2 = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, items2);
        dropdown2.setAdapter(adapter2);
        ArrayAdapter<String> adapter3 = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, items3);
        dropdown3.setAdapter(adapter3);
        ArrayAdapter<String> adapter4 = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, items4);
        dropdown4.setAdapter(adapter4);
        ArrayAdapter<String> adapter5 = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, items5);
        dropdown5.setAdapter(adapter5);


        s1.setOnItemSelectedListener(this);
        s2.setOnItemSelectedListener(this);
        s3.setOnItemSelectedListener(this);
        s4.setOnItemSelectedListener(this);
        s5.setOnItemSelectedListener(this);
        s6.setOnItemSelectedListener(this);
    }

    public void submit(View view) {

        pregnancies=e1.getText().toString();
        skinthickness=e2.getText().toString();

       /* int GlucoseId= (int) s1.getSelectedItemId();
        final Spinner Glucosebtn=findViewById(GlucoseId);

        int BloodPressureId= (int) s2.getSelectedItemId();
        Spinner BloodPressurebtn=findViewById(BloodPressureId);

        int InsulinId= (int) s3.getSelectedItemId();
        Spinner Insulinbtn=findViewById(InsulinId);

        int BmiId= (int) s4.getSelectedItemId();
        Spinner Bmibtn=findViewById(BmiId);

        int DpfId= (int) s5.getSelectedItemId();
        Spinner Dpfbtn=findViewById(DpfId);

        int AgeId= (int) s6.getSelectedItemId();
        Spinner Agebtn=findViewById(AgeId);


        if(Glucosebtn.getSelectedItem() != null){
            glucose=Glucosebtn.getSelectedItem().toString();}
        if(Insulinbtn.getSelectedItem() != null){
            insulin=Insulinbtn.getSelectedItem().toString();}
        if(Bmibtn.getSelectedItem() != null){
            bmi=Bmibtn.getSelectedItem().toString();}
        if(Dpfbtn.getSelectedItem() != null){
            DPF=Dpfbtn.getSelectedItem().toString();}
        if(BloodPressurebtn.getSelectedItem() != null){
            bp=BloodPressurebtn.getSelectedItem().toString();}
        if(Agebtn.getSelectedItem() != null){
            age=Agebtn.getSelectedItem().toString();}
*/
        if (awesomeValidation.validate()) {
            if(!TextUtils.isEmpty(pregnancies)&&(!TextUtils.isEmpty(skinthickness)))
            {
                RequestQueue requestQueue = Volley.newRequestQueue(this);
                final String url = "";
                JSONObject postParams = new JSONObject();
                try {
                    postParams.put("Pregnancies", pregnancies);
                    postParams.put("Glucose", glucose);
                    postParams.put("BloodPressure", bp);
                    postParams.put("SkinThickness", skinthickness);
                    postParams.put("Insulin", insulin);
                    postParams.put("BMI", bmi);
                    postParams.put("DiabeticPedigreeFunction", DPF);
                    postParams.put("Age", age);
                } catch (JSONException e)
                {
                    e.printStackTrace();
                }
                JsonObjectRequest jsonObjectRequest=new JsonObjectRequest(Request.Method.POST, url, postParams, new Response.Listener<JSONObject>() {
                    @Override
                    public void onResponse(JSONObject response) {
                        Log.i("On Response", "onResponse: " + response.toString());
                    }
                }, new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        Log.i("On Error",error.toString());
                        //Toast.makeText(MainActivity.this, ""+error.toString(), Toast.LENGTH_SHORT).show();
                    }
                });
                requestQueue.add(jsonObjectRequest);
            }
            Intent i=new Intent(MainActivity2.this,ResultsActivity.class);
            startActivity(i);
        }
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        String item = parent.getItemAtPosition(position).toString();
        i=i+1;
        if(i==1){
        glucose=item;}
        if(i==2){
            insulin=item;}
        if(i==3){
            bmi=item;}
        if(i==4){
            DPF=item;}
        if(i==5){
            bp=item;}
        if(i==6){
            age=item;}
        // Showing selected spinner item
        //Toast.makeText(parent.getContext(), "Selected: " + item, Toast.LENGTH_LONG).show();
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

    }
}