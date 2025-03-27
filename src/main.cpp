#include "Renderer.h"

int main()
{
    int W = 500, H = 500;
    try 
    { 
        Renderer renderer(H, W);
        renderer.render();
    }
    catch (const std::runtime_error& e) { std::cerr << "Exception thrown:\n" << e.what() << std::endl; }
  
    return 0;
}
