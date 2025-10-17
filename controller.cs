using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Text.Json;
using System.IO;

namespace AspireVisionLauncher.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly IConfiguration _configuration;

        // Configuration for Python script execution
        private readonly string _pythonExecutable;
        private readonly string _scriptPath;
        private readonly string _workingDirectory;
        private readonly string _processName;
        
        // Simple process tracking
        private static Process? _dashboardProcess = null;

        public HomeController(ILogger<HomeController> logger, IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;
            
            // Get configuration values for Python execution
            _pythonExecutable = _configuration["AspireVision:ExecutablePath"] ?? "pythonw";
            _scriptPath = _configuration["AspireVision:Arguments"] ?? @"C:\Aspire Dashboard\start_dashboard.pyw";
            _workingDirectory = _configuration["AspireVision:WorkingDirectory"] ?? @"C:\Aspire Dashboard";
            _processName = _configuration["AspireVision:ProcessName"] ?? "pythonw";
        }

        public IActionResult Index()
        {
            // Check if Python is available and script exists
            var pythonAvailable = IsPythonAvailable();
            var scriptExists = System.IO.File.Exists(_scriptPath);
            
            var model = new DashboardViewModel
            {
                ExecutablePath = $"{_pythonExecutable} {_scriptPath}",
                IsExecutableAvailable = pythonAvailable && scriptExists
            };

            return View(model);
        }
        
        private bool IsPythonAvailable()
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = _pythonExecutable,
                        Arguments = "--version",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        WindowStyle = ProcessWindowStyle.Hidden
                    }
                };
                
                process.Start();
                process.WaitForExit(5000); // Wait max 5 seconds
                return process.ExitCode == 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to check Python availability");
                return false;
            }
        }
        
        private List<Process> GetDashboardProcesses()
        {
            var dashboardProcesses = new List<Process>();
            
            try
            {
                // Check if we have a tracked process that's still running
                if (_dashboardProcess != null && !_dashboardProcess.HasExited)
                {
                    dashboardProcesses.Add(_dashboardProcess);
                    return dashboardProcesses;
                }
                
                // Fallback: Look for Python/pythonw processes (simplified approach)
                var pythonProcesses = Process.GetProcessesByName("pythonw").Union(Process.GetProcessesByName("python")).ToList();
                foreach (var process in pythonProcesses)
                {
                    try
                    {
                        // Simple heuristic: check if process was started recently and has our working directory
                        if (process.StartTime > DateTime.Now.AddHours(-1)) // Started within last hour
                        {
                            dashboardProcesses.Add(process);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogDebug(ex, $"Could not check process {process.Id}");
                    }
                }
                
                return dashboardProcesses;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting dashboard processes");
                return new List<Process>();
            }
        }

        [HttpPost]
        public async Task<IActionResult> LaunchDashboard()
        {
            try
            {
                _logger.LogInformation("Attempting to launch AspireVision Dashboard...");

                // Check if Python is available
                if (!IsPythonAvailable())
                {
                    var error = "Python not found or not working. Please ensure Python is installed and accessible.";
                    _logger.LogError(error);
                    return Json(new { success = false, message = error });
                }

                // Check if script file exists
                if (!System.IO.File.Exists(_scriptPath))
                {
                    var error = $"Python script not found at: {_scriptPath}";
                    _logger.LogError(error);
                    return Json(new { success = false, message = error });
                }

                // Check if the dashboard is already running by looking for processes running the script
                var existingProcesses = GetDashboardProcesses();
                if (existingProcesses.Any())
                {
                    var warning = "AspireVision Dashboard is already running.";
                    _logger.LogWarning(warning);
                    return Json(new { success = false, message = warning });
                }

                // Start the Python script
                var processStartInfo = new ProcessStartInfo
                {
                    FileName = _pythonExecutable,
                    Arguments = $"\"{_scriptPath}\"",
                    WorkingDirectory = _workingDirectory,
                    UseShellExecute = true,
                    CreateNoWindow = true,
                    WindowStyle = ProcessWindowStyle.Hidden
                };

                var process = Process.Start(processStartInfo);
                
                // Track the process we started
                _dashboardProcess = process;
                
                if (process != null)
                {
                    _logger.LogInformation($"Successfully launched AspireVision Dashboard with PID: {process.Id}");
                    
                    // Wait a moment to see if the process starts successfully
                    await Task.Delay(2000);
                    
                    if (!process.HasExited)
                    {
                        return Json(new { 
                            success = true, 
                            message = "AspireVision Dashboard launched successfully!",
                            processId = process.Id
                        });
                    }
                    else
                    {
                        var exitError = $"Process exited immediately with code: {process.ExitCode}";
                        _logger.LogError(exitError);
                        return Json(new { success = false, message = exitError });
                    }
                }
                else
                {
                    var error = "Failed to start the process";
                    _logger.LogError(error);
                    return Json(new { success = false, message = error });
                }
            }
            catch (Exception ex)
            {
                var error = $"Error launching dashboard: {ex.Message}";
                _logger.LogError(ex, error);
                return Json(new { success = false, message = error });
            }
        }

        [HttpPost]
        public IActionResult StopDashboard()
        {
            try
            {
                _logger.LogInformation("Attempting to stop AspireVision Dashboard...");

                var processes = GetDashboardProcesses();
                var stoppedCount = 0;

                foreach (var process in processes)
                {
                    try
                    {
                        _logger.LogInformation($"Stopping dashboard process with PID: {process.Id}");
                        process.Kill();
                        stoppedCount++;
                        
                        // Clear tracked process if we stopped it
                        if (_dashboardProcess?.Id == process.Id)
                        {
                            _dashboardProcess = null;
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, $"Failed to stop process with PID: {process.Id}");
                    }
                    finally
                    {
                        process.Dispose();
                    }
                }

                if (stoppedCount > 0)
                {
                    return Json(new { 
                        success = true, 
                        message = $"Stopped {stoppedCount} AspireVision Dashboard process(es)."
                    });
                }
                else
                {
                    return Json(new { 
                        success = false, 
                        message = "No AspireVision Dashboard processes found running."
                    });
                }
            }
            catch (Exception ex)
            {
                var error = $"Error stopping dashboard: {ex.Message}";
                _logger.LogError(ex, error);
                return Json(new { success = false, message = error });
            }
        }

        [HttpGet]
        public IActionResult GetStatus()
        {
            try
            {
                var processes = GetDashboardProcesses();
                var isRunning = processes.Count > 0;
                var processDetails = processes.Select(p => new { 
                    Id = p.Id, 
                    StartTime = p.StartTime,
                    ProcessName = p.ProcessName
                }).ToArray();

                // Dispose of process objects
                foreach (var process in processes)
                {
                    process.Dispose();
                }

                return Json(new { 
                    isRunning = isRunning,
                    processCount = processDetails.Length,
                    processes = processDetails
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking dashboard status");
                return Json(new { isRunning = false, error = ex.Message });
            }
        }

        public IActionResult Error()
        {
            return View();
        }
    }

    public class DashboardViewModel
    {
        public string ExecutablePath { get; set; } = string.Empty;
        public bool IsExecutableAvailable { get; set; }
    }
} 
