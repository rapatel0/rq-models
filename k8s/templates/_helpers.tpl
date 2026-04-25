{{/*
Expand the name of the chart.
*/}}
{{- define "rotorquant.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "rotorquant.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "rotorquant.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "rotorquant.labels" -}}
helm.sh/chart: {{ include "rotorquant.chart" . }}
{{ include "rotorquant.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "rotorquant.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rotorquant.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "rotorquant.modelsPvcName" -}}
{{- if .Values.models.existingClaim -}}
{{- .Values.models.existingClaim -}}
{{- else -}}
{{- printf "%s-models" (include "rotorquant.fullname" .) -}}
{{- end -}}
{{- end -}}
