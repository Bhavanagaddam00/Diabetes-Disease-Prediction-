package com.example.diabetesdiseaseprediction;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;

import com.basgeekball.awesomevalidation.AwesomeValidation;
import com.basgeekball.awesomevalidation.ValidationStyle;
import com.google.common.collect.Range;

public class MainActivity2 extends AppCompatActivity {

    EditText e1,e2;
    Button s;
    AwesomeValidation awesomeValidation;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);
        e1=findViewById(R.id.e1);
        e2=findViewById(R.id.e2);
        s=findViewById(R.id.s);

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


    }

    public void submit(View view) {

        if (awesomeValidation.validate()) {
            Intent i=new Intent(MainActivity2.this,ResultsActivity.class);
            startActivity(i);
        }
    }
}