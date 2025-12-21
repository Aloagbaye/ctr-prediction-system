# PowerShell Tutorial

## 📋 Overview

This tutorial provides a comprehensive guide to PowerShell scripting, covering the fundamentals you need to understand and write PowerShell scripts like those used in the CTR Prediction System deployment.

**Prerequisites:**
- Windows 10/11 or Windows Server
- PowerShell 5.1+ (or PowerShell Core 7+)
- Basic understanding of command-line interfaces

---

## 🚀 Getting Started

### What is PowerShell?

PowerShell is a task automation and configuration management framework from Microsoft, consisting of a command-line shell and scripting language. It's built on the .NET Framework and provides powerful features for system administration and automation.

### Opening PowerShell

**Windows:**
- Press `Win + X` and select "Windows PowerShell" or "Terminal"
- Search for "PowerShell" in the Start menu
- Right-click Start button → "Windows PowerShell"

**Check Version:**
```powershell
$PSVersionTable
```

---

## 📝 Basics

### Comments

```powershell
# Single-line comment

<#
    Multi-line comment
    Can span multiple lines
#>
```

### Output

```powershell
# Print to console
Write-Host "Hello, World!"

# Print with color
Write-Host "Success!" -ForegroundColor Green
Write-Host "Warning!" -ForegroundColor Yellow
Write-Host "Error!" -ForegroundColor Red

# Print without newline
Write-Host "Loading..." -NoNewline
```

### Execution Policy

PowerShell has an execution policy that controls script execution:

```powershell
# Check current policy
Get-ExecutionPolicy

# Set policy (requires admin)
Set-ExecutionPolicy RemoteSigned
# Options: Restricted, AllSigned, RemoteSigned, Unrestricted
```

---

## 🔢 Variables

### Variable Basics

```powershell
# Create a variable
$name = "John"
$age = 30
$isActive = $true

# Display variable
Write-Host $name
Write-Host "Name: $name, Age: $age"

# Variable in string (interpolation)
$message = "Hello, $name!"
```

### Variable Types

```powershell
# String
$text = "Hello"
$text = 'Hello'  # Single quotes (no interpolation)

# Integer
$number = 42
$number = [int]42  # Explicit type

# Boolean
$flag = $true
$flag = $false

# Array
$items = @("apple", "banana", "cherry")
$numbers = 1, 2, 3, 4, 5

# Hash table (dictionary)
$person = @{
    Name = "John"
    Age = 30
    City = "New York"
}
```

### Variable Scope

```powershell
# Global scope
$global:globalVar = "I'm global"

# Script scope
$script:scriptVar = "I'm in script"

# Local scope (default)
$localVar = "I'm local"

# Function scope
function Test-Function {
    $functionVar = "I'm in function"
}
```

### Special Variables

```powershell
# Current directory
$PWD
Get-Location

# Home directory
$HOME

# Error action preference
$ErrorActionPreference  # Stop, Continue, SilentlyContinue

# Last exit code
$LASTEXITCODE

# Environment variables
$env:USERNAME
$env:PATH
$env:GCP_PROJECT_ID = "my-project"  # Set environment variable
```

### Variable Operations

```powershell
# Concatenation
$first = "Hello"
$second = "World"
$combined = $first + " " + $second

# Type conversion
$number = "42"
$intNumber = [int]$number
$stringNumber = [string]$intNumber

# Check if variable exists
if ($null -ne $myVar) {
    Write-Host "Variable exists"
}

# Remove variable
Remove-Variable name
$name = $null
```

---

## 🔀 Conditional Statements

### If-Else Statements

```powershell
# Basic if
if ($condition) {
    Write-Host "Condition is true"
}

# If-else
if ($age -ge 18) {
    Write-Host "Adult"
} else {
    Write-Host "Minor"
}

# If-elseif-else
if ($score -ge 90) {
    Write-Host "Grade: A"
} elseif ($score -ge 80) {
    Write-Host "Grade: B"
} elseif ($score -ge 70) {
    Write-Host "Grade: C"
} else {
    Write-Host "Grade: F"
}
```

### Comparison Operators

```powershell
# Equality
$value -eq 10      # Equal
$value -ne 10      # Not equal

# Comparison
$value -gt 10      # Greater than
$value -ge 10      # Greater than or equal
$value -lt 10      # Less than
$value -le 10      # Less than or equal

# String comparison
$text -eq "hello"   # Case-insensitive
$text -ceq "Hello"  # Case-sensitive
$text -like "he*"   # Wildcard match
$text -match "regex" # Regex match

# Containment
$array -contains "item"    # Array contains item
$array -notcontains "item" # Array doesn't contain item
$text -in $array           # Item in array
```

### Logical Operators

```powershell
# AND
if ($age -ge 18 -and $hasLicense) {
    Write-Host "Can drive"
}

# OR
if ($status -eq "active" -or $status -eq "pending") {
    Write-Host "Valid status"
}

# NOT
if (-not $isComplete) {
    Write-Host "Not complete"
}

# Alternative syntax
if (!$isComplete) {
    Write-Host "Not complete"
}
```

### Switch Statements

```powershell
# Basic switch
switch ($day) {
    "Monday"    { Write-Host "Start of week" }
    "Friday"    { Write-Host "End of week" }
    "Saturday"  { Write-Host "Weekend!" }
    "Sunday"    { Write-Host "Weekend!" }
    default     { Write-Host "Weekday" }
}

# Switch with conditions
switch ($score) {
    { $_ -ge 90 } { Write-Host "A" }
    { $_ -ge 80 } { Write-Host "B" }
    { $_ -ge 70 } { Write-Host "C" }
    default       { Write-Host "F" }
}

# Switch with multiple matches
switch -Wildcard ($filename) {
    "*.txt"  { Write-Host "Text file" }
    "*.ps1"  { Write-Host "PowerShell script" }
    "*.json" { Write-Host "JSON file" }
}
```

### Ternary Operator (PowerShell 7+)

```powershell
# Traditional if-else
if ($condition) {
    $result = "Yes"
} else {
    $result = "No"
}

# Ternary (PowerShell 7+)
$result = $condition ? "Yes" : "No"
```

---

## 🔧 Functions

### Basic Functions

```powershell
# Simple function
function Say-Hello {
    Write-Host "Hello, World!"
}

# Call function
Say-Hello

# Function with parameters
function Greet-Person {
    param(
        [string]$Name
    )
    Write-Host "Hello, $Name!"
}

Greet-Person -Name "John"
Greet-Person "John"  # Positional parameter
```

### Function Parameters

```powershell
# Multiple parameters
function Add-Numbers {
    param(
        [int]$First,
        [int]$Second
    )
    return $First + $Second
}

$sum = Add-Numbers -First 5 -Second 3

# Parameters with defaults
function Create-User {
    param(
        [string]$Username,
        [string]$Role = "User",
        [switch]$IsActive
    )
    Write-Host "User: $Username, Role: $Role, Active: $IsActive"
}

Create-User -Username "john" -IsActive
Create-User "jane" -Role "Admin"
```

### Parameter Types

```powershell
function Process-Data {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Name,
        
        [int]$Age = 0,
        
        [ValidateSet("Active", "Inactive")]
        [string]$Status = "Active",
        
        [switch]$Force,
        
        [string[]]$Tags
    )
    # Function body
}
```

### Return Values

```powershell
# Explicit return
function Get-Square {
    param([int]$Number)
    return $Number * $Number
}

# Implicit return (last value)
function Get-Square {
    param([int]$Number)
    $Number * $Number
}

# Multiple return values
function Get-UserInfo {
    param([string]$Username)
    return @{
        Name = $Username
        Age = 30
        Active = $true
    }
}
```

### Advanced Functions

```powershell
# Function with error handling
function Test-Connection {
    param([string]$Host)
    
    try {
        Test-Connection -ComputerName $Host -Count 1 -ErrorAction Stop
        return $true
    } catch {
        Write-Error "Failed to connect to $Host"
        return $false
    }
}

# Function with pipeline input
function Get-FileSize {
    param(
        [Parameter(ValueFromPipeline=$true)]
        [string]$Path
    )
    
    process {
        if (Test-Path $Path) {
            $file = Get-Item $Path
            [PSCustomObject]@{
                Path = $Path
                Size = $file.Length
            }
        }
    }
}

# Usage
Get-ChildItem *.txt | Get-FileSize
```

### Function Examples from Scripts

```powershell
# Example from gcp_deploy.ps1
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Check-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
        Write-Error "gcloud CLI is not installed"
        exit 1
    }
    
    if ([string]::IsNullOrEmpty($PROJECT_ID)) {
        Write-Error "PROJECT_ID not set"
        exit 1
    }
}
```

---

## 🔄 Loops

### For Loop

```powershell
# Basic for loop
for ($i = 1; $i -le 10; $i++) {
    Write-Host $i
}

# Loop through array
$items = @("apple", "banana", "cherry")
for ($i = 0; $i -lt $items.Length; $i++) {
    Write-Host $items[$i]
}
```

### ForEach Loop

```powershell
# ForEach loop
$items = @("apple", "banana", "cherry")
foreach ($item in $items) {
    Write-Host $item
}

# ForEach-Object (pipeline)
1..10 | ForEach-Object {
    Write-Host $_
}

# ForEach-Object with property
Get-ChildItem | ForEach-Object {
    Write-Host $_.Name
}
```

### While Loop

```powershell
# While loop
$counter = 0
while ($counter -lt 10) {
    Write-Host $counter
    $counter++
}

# Do-While loop
$counter = 0
do {
    Write-Host $counter
    $counter++
} while ($counter -lt 10)
```

### Loop Control

```powershell
# Break (exit loop)
for ($i = 1; $i -le 10; $i++) {
    if ($i -eq 5) {
        break  # Exit loop
    }
    Write-Host $i
}

# Continue (skip iteration)
for ($i = 1; $i -le 10; $i++) {
    if ($i -eq 5) {
        continue  # Skip this iteration
    }
    Write-Host $i
}
```

---

## 📁 File Operations

### Working with Files

```powershell
# Check if file exists
if (Test-Path "file.txt") {
    Write-Host "File exists"
}

# Check if directory exists
if (Test-Path "directory" -PathType Container) {
    Write-Host "Directory exists"
}

# Read file
$content = Get-Content "file.txt"
$content = Get-Content "file.txt" -Raw  # As single string

# Write file
Set-Content "file.txt" -Value "Hello, World!"
"Hello, World!" | Out-File "file.txt"

# Append to file
Add-Content "file.txt" -Value "New line"

# Copy file
Copy-Item "source.txt" -Destination "dest.txt"

# Move file
Move-Item "source.txt" -Destination "dest.txt"

# Delete file
Remove-Item "file.txt"
```

### Working with Directories

```powershell
# List files
Get-ChildItem
Get-ChildItem -Path "C:\Users" -Recurse

# Filter files
Get-ChildItem *.txt
Get-ChildItem -Filter "*.pkl"

# Create directory
New-Item -ItemType Directory -Path "newdir"
mkdir "newdir"  # Alias

# Remove directory
Remove-Item "directory" -Recurse
rmdir "directory"  # Alias
```

---

## 🔍 Common Commands

### Get-Command

```powershell
# List all commands
Get-Command

# Find command
Get-Command *process*
Get-Command Get-*

# Get command details
Get-Command Get-Process | Format-List
```

### Get-Help

```powershell
# Get help for command
Get-Help Get-Process
Get-Help Get-Process -Examples
Get-Help Get-Process -Full
Get-Help Get-Process -Online
```

### Error Handling

```powershell
# Try-Catch
try {
    $result = 10 / 0
} catch {
    Write-Host "Error: $_"
    Write-Host "Error type: $($_.Exception.GetType().FullName)"
}

# Try-Catch-Finally
try {
    # Code that might fail
} catch {
    # Handle error
} finally {
    # Always executes
    # Cleanup code
}

# ErrorAction parameter
Get-Item "nonexistent.txt" -ErrorAction SilentlyContinue
Get-Item "nonexistent.txt" -ErrorAction Stop
Get-Item "nonexistent.txt" -ErrorAction Continue
```

### Pipeline

```powershell
# Chain commands
Get-Process | Where-Object {$_.CPU -gt 100} | Sort-Object CPU -Descending

# Filter with Where-Object
Get-ChildItem | Where-Object {$_.Length -gt 1MB}

# Select properties
Get-Process | Select-Object Name, CPU, Memory

# Sort
Get-ChildItem | Sort-Object Length -Descending

# Group
Get-ChildItem | Group-Object Extension
```

### String Operations

```powershell
# String methods
$text = "Hello World"
$text.ToUpper()        # "HELLO WORLD"
$text.ToLower()        # "hello world"
$text.Length           # 11
$text.Substring(0, 5) # "Hello"
$text.Replace("World", "PowerShell")  # "Hello PowerShell"

# String formatting
$name = "John"
$age = 30
"Name: {0}, Age: {1}" -f $name, $age

# String splitting
$text = "apple,banana,cherry"
$items = $text -split ","
```

### Arrays and Collections

```powershell
# Create array
$array = @(1, 2, 3, 4, 5)
$array = 1..5  # Range

# Access elements
$array[0]      # First element
$array[-1]     # Last element
$array[0..2]   # First three elements

# Add to array
$array += 6
$array = $array + 7

# Array methods
$array.Count
$array.Length
$array.Contains(3)
$array.IndexOf(3)

# Hash table
$hash = @{
    Name = "John"
    Age = 30
}
$hash.Name
$hash["Name"]
$hash.Keys
$hash.Values
```

### Date and Time

```powershell
# Get current date/time
Get-Date
$now = Get-Date

# Format date
Get-Date -Format "yyyy-MM-dd"
Get-Date -Format "yyyyMMdd"

# Date arithmetic
$date = Get-Date
$date.AddDays(7)
$date.AddMonths(1)
$date.AddYears(1)
```

### Environment Variables

```powershell
# Get environment variable
$env:USERNAME
$env:PATH
$env:GCP_PROJECT_ID

# Set environment variable (session)
$env:MY_VAR = "value"

# Set environment variable (permanent)
[Environment]::SetEnvironmentVariable("MY_VAR", "value", "User")
```

---

## 🛠️ Advanced Topics

### Error Action Preference

```powershell
# Set error handling behavior
$ErrorActionPreference = "Stop"      # Stop on error
$ErrorActionPreference = "Continue"  # Continue on error
$ErrorActionPreference = "SilentlyContinue"  # Suppress errors

# Per-command
Get-Item "file.txt" -ErrorAction SilentlyContinue
```

### Output Redirection

```powershell
# Redirect output
Get-Process > output.txt
Get-Process >> output.txt  # Append

# Redirect errors
Get-Item "file.txt" 2> errors.txt

# Redirect both
Get-Process * 2>&1 > output.txt

# Suppress output
Get-Process | Out-Null
```

### Splatting

```powershell
# Instead of this:
Get-ChildItem -Path "C:\Users" -Recurse -Filter "*.txt" -ErrorAction SilentlyContinue

# Use splatting:
$params = @{
    Path = "C:\Users"
    Recurse = $true
    Filter = "*.txt"
    ErrorAction = "SilentlyContinue"
}
Get-ChildItem @params
```

### Here-Strings

```powershell
# Multi-line string
$text = @"
This is a
multi-line
string
"@

# Here-string with variables
$name = "John"
$message = @"
Hello $name,
This is a message.
"@
```

### Modules

```powershell
# List modules
Get-Module -ListAvailable

# Import module
Import-Module ModuleName

# Check if module is loaded
Get-Module ModuleName
```

---

## 💡 Best Practices

### 1. Use Meaningful Variable Names

```powershell
# Bad
$x = "John"
$n = 30

# Good
$userName = "John"
$userAge = 30
```

### 2. Use Functions for Reusability

```powershell
# Bad: Repeated code
Write-Host "[INFO] Starting..."
Write-Host "[INFO] Processing..."
Write-Host "[INFO] Complete!"

# Good: Function
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

Write-Info "Starting..."
Write-Info "Processing..."
Write-Info "Complete!"
```

### 3. Handle Errors Properly

```powershell
# Bad: No error handling
$file = Get-Content "file.txt"

# Good: Error handling
try {
    $file = Get-Content "file.txt" -ErrorAction Stop
} catch {
    Write-Error "Failed to read file: $_"
    exit 1
}
```

### 4. Use Parameters Instead of Global Variables

```powershell
# Bad: Global variable
$PROJECT_ID = "my-project"
function Deploy {
    Write-Host "Deploying to $PROJECT_ID"
}

# Good: Parameter
function Deploy {
    param([string]$ProjectId)
    Write-Host "Deploying to $ProjectId"
}
```

### 5. Check Prerequisites

```powershell
# Check if command exists
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "gcloud not found"
    exit 1
}

# Check if file exists
if (-not (Test-Path "config.json")) {
    Write-Error "config.json not found"
    exit 1
}
```

---

## 🎯 Common Patterns from Deployment Scripts

### Pattern 1: Configuration Variables

```powershell
# Set configuration at top of script
$PROJECT_ID = if ($env:GCP_PROJECT_ID) { $env:GCP_PROJECT_ID } else { "" }
$REGION = if ($env:GCP_REGION) { $env:GCP_REGION } else { "us-central1" }
$RESOURCE_PREFIX = "ctr-prediction"
```

### Pattern 2: Helper Functions

```powershell
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}
```

### Pattern 3: Check Resource Existence

```powershell
# Check if bucket exists
$bucketExists = $false
try {
    $null = gsutil ls -b "gs://${BUCKET_NAME}" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $bucketExists = $true
    }
} catch {
    $bucketExists = $false
}

if ($bucketExists) {
    Write-Warn "Bucket already exists"
} else {
    Write-Info "Creating bucket..."
}
```

### Pattern 4: Error Handling with Exit Codes

```powershell
# Run command and check exit code
$output = gsutil -m -q cp *.pkl "gs://${BUCKET_NAME}/" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Upload may have failed"
} else {
    Write-Info "Upload successful"
}
```

---

## 📚 Additional Resources

- [PowerShell Documentation](https://docs.microsoft.com/powershell/)
- [PowerShell Learning Path](https://docs.microsoft.com/learn/paths/powershell/)
- [PowerShell Gallery](https://www.powershellgallery.com/)
- [PowerShell GitHub](https://github.com/PowerShell/PowerShell)

---

## 🎓 Quick Reference

### Essential Commands

```powershell
# Variables
$var = "value"
$env:VAR = "value"

# Conditionals
if ($condition) { }
elseif ($condition) { }
else { }

# Functions
function Name { param($param) }

# Loops
foreach ($item in $items) { }
for ($i = 0; $i -lt 10; $i++) { }
while ($condition) { }

# Files
Test-Path "file.txt"
Get-Content "file.txt"
Set-Content "file.txt" -Value "text"

# Error handling
try { } catch { } finally { }
```

---

**Happy Scripting!** 🚀

