import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Sign In - Ummid Se Hari',
  description: 'Sign in to your account',
};

export default function SignInPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4 py-12 sm:px-6 lg:px-8">
      <div className="w-full max-w-md space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Sign in to your account
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            We&apos;ll send you a secure sign-in link via email
          </p>
        </div>
        {/* SignIn form component will go here in next commits */}
        <div className="mt-8">
          <div className="bg-white px-4 py-8 shadow sm:rounded-lg sm:px-10">
            <p className="text-center text-gray-500">
              Sign-in form will be implemented in the next phase
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
