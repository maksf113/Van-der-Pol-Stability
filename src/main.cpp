#include "Application.h"

int main()
{
    constexpr int W = 500;
    constexpr int H = 500;
    try 
    { 
        Application app(W, H);
        app.run();
    }
    catch (const std::runtime_error& e) { std::cerr << "Exception thrown:\n" << e.what() << std::endl; }
  
    return 0;
}
