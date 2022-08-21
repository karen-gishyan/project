# Generated by Django 4.1 on 2022-08-20 20:16

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('mimic3', '0003_alter_prescriptions_route'),
    ]

    operations = [
        migrations.CreateModel(
            name='D_Items',
            fields=[
                ('row_id', models.IntegerField()),
                ('itemid', models.IntegerField(primary_key=True, serialize=False)),
                ('label', models.CharField(blank=True, max_length=100, null=True)),
                ('abbreviation', models.CharField(blank=True, max_length=100, null=True)),
                ('dbsource', models.CharField(blank=True, max_length=100, null=True)),
                ('linksto', models.CharField(blank=True, max_length=100, null=True)),
                ('category', models.CharField(blank=True, max_length=100, null=True)),
                ('unitname', models.CharField(blank=True, max_length=100, null=True)),
                ('param_type', models.CharField(blank=True, max_length=100, null=True)),
                ('conceptid', models.CharField(blank=True, max_length=100, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='D_LabItems',
            fields=[
                ('row_id', models.IntegerField()),
                ('itemid', models.IntegerField(primary_key=True, serialize=False)),
                ('label', models.CharField(blank=True, max_length=100, null=True)),
                ('fluid', models.CharField(blank=True, max_length=100, null=True)),
                ('category', models.CharField(blank=True, max_length=100, null=True)),
                ('loinc_code', models.CharField(blank=True, max_length=100, null=True)),
            ],
        ),
        migrations.RemoveField(
            model_name='inputevent_mv',
            name='statusdescription',
        ),
        migrations.AddField(
            model_name='inputevent_mv',
            name='patient_weight',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True),
        ),
        migrations.CreateModel(
            name='LabEvents',
            fields=[
                ('row_id', models.IntegerField(primary_key=True, serialize=False)),
                ('charttime', models.DateTimeField(blank=True, null=True)),
                ('value', models.CharField(blank=True, max_length=100, null=True)),
                ('valuenum', models.DecimalField(blank=True, decimal_places=3, max_digits=100, null=True)),
                ('valueuom', models.CharField(blank=True, max_length=100, null=True)),
                ('flag', models.CharField(blank=True, max_length=100, null=True)),
                ('hadm_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='mimic3.admissions')),
                ('itemid', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mimic3.d_labitems')),
                ('subject_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mimic3.patients')),
            ],
        ),
        migrations.CreateModel(
            name='ChartEvents',
            fields=[
                ('row_id', models.IntegerField(primary_key=True, serialize=False)),
                ('charttime', models.DateTimeField(blank=True, null=True)),
                ('storetime', models.DateTimeField(blank=True, null=True)),
                ('cgid', models.PositiveIntegerField(blank=True, null=True)),
                ('value', models.CharField(blank=True, max_length=100, null=True)),
                ('valuenum', models.DecimalField(blank=True, decimal_places=3, max_digits=100, null=True)),
                ('valueuom', models.CharField(blank=True, max_length=100, null=True)),
                ('warning', models.PositiveIntegerField(blank=True, null=True)),
                ('error', models.PositiveIntegerField(blank=True, null=True)),
                ('resultstatus', models.CharField(blank=True, max_length=100, null=True)),
                ('stopped', models.CharField(blank=True, max_length=100, null=True)),
                ('hadm_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mimic3.admissions')),
                ('icustay_id', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='mimic3.icustays')),
                ('itemid', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mimic3.d_items')),
                ('subject_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mimic3.patients')),
            ],
        ),
    ]
